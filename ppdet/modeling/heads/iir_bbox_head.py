# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.
# ------------------------------------------------------------------------
# Additionally modified by Yunqiu Xu for H2FA R-CNN
# ------------------------------------------------------------------------

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform, KaimingNormal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta
from ppdet.modeling.layers import ConvNormLayer

__all__ = ['IIRBBoxHead']


@register
class IIRBBoxHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner', 'bbox_loss']
    """
    RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        with_pool (bool): Whether to use pooling for the RoI feature.
        num_classes (int): The number of classes
        bbox_weight (List[float]): The weight to get the decode box 
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.],
                 bbox_loss=None):
        super(IIRBBoxHead, self).__init__()
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner

        self.with_pool = with_pool
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.bbox_loss = bbox_loss

        self.bbox_score = nn.Linear(
            in_channel,
            self.num_classes + 1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)))
        self.bbox_score.skip_quant = True

        self.bbox_delta = nn.Linear(
            in_channel,
            4 * self.num_classes,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.001)))
        self.bbox_delta.skip_quant = True
        self.assigned_label = None
        self.assigned_rois = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape[0].channels
        }
    
    def forward_weak(self, body_feats, rois, rois_num, inputs, obj_logits, gt_classes_img_oh):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        assert self.training, "forward_weak only used in training"

        sampled_rois = []
        sampled_obj_logits = []
        num_sampled_roi = []
        for rois_per_image, obj_logits_per_image in zip(rois, obj_logits):
            sampled_idxs = paddle.randperm(len(rois_per_image))[:self.bbox_assigner.batch_size_per_im]
            sampled_rois.append(rois_per_image[sampled_idxs])
            sampled_obj_logits.append(obj_logits_per_image[sampled_idxs])
            num_sampled_roi.append(len(sampled_idxs))

        # TODO: better implementation to calculate loss using rest inputs with non-valid proposals
        if 0 in num_sampled_roi:
            return {'loss_img_cls': 0.0 * body_feats[0].sum()}

        rois_feat = self.roi_extractor(
            body_feats, sampled_rois, paddle.to_tensor(num_sampled_roi, dtype=paddle.int32)
        )
        bbox_feat = self.head(rois_feat)
        scores = self.bbox_score(bbox_feat)
        cls_scores = F.softmax(scores[:,:-1], axis=1)

        sampled_obj_logits = paddle.concat(sampled_obj_logits).unsqueeze(-1)

        max_cls_ids = paddle.argmax(cls_scores, axis=1)

        # TODO: better implementation with higher efficiency, as `torch.Tensor.scatter_()'
        obj_scores = F.one_hot(max_cls_ids, self.num_classes) * sampled_obj_logits.tile([1, self.num_classes])

        pred_img_cls_logits = paddle.concat(
            [
                paddle.sum(cls*F.softmax(obj, axis=0), axis=0, keepdim=True)
                for cls, obj in zip(
                cls_scores.split(num_sampled_roi, axis=0),
                obj_scores.split(num_sampled_roi, axis=0))
            ], axis=0
        )

        loss = F.binary_cross_entropy(
            paddle.clip(pred_img_cls_logits, min=1e-6, max=1.0 - 1e-6),
            gt_classes_img_oh,
            reduction='mean'
        )
        return {'loss_img_cls': loss}
    
    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if self.with_pool:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(scores, deltas, targets, rois,
                                 self.bbox_weight)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, self.head

    def get_loss(self, scores, deltas, targets, rois, bbox_weight):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}

        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets

        # bbox cls
        tgt_labels = paddle.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        valid_inds = paddle.nonzero(tgt_labels >= 0).flatten()
        if valid_inds.shape[0] == 0:
            loss_bbox[cls_name] = paddle.zeros([1], dtype='float32')
        else:
            tgt_labels = tgt_labels.cast('int64')
            tgt_labels.stop_gradient = True
            loss_bbox_cls = F.cross_entropy(
                input=scores, label=tgt_labels, reduction='mean')
            loss_bbox[cls_name] = loss_bbox_cls

        # bbox reg
        cls_agnostic_bbox_reg = deltas.shape[1] == 4

        fg_inds = paddle.nonzero(
            paddle.logical_and(tgt_labels >= 0, tgt_labels <
                               self.num_classes)).flatten()

        if fg_inds.numel() == 0:
            loss_bbox[reg_name] = paddle.zeros([1], dtype='float32')
            return loss_bbox

        if cls_agnostic_bbox_reg:
            reg_delta = paddle.gather(deltas, fg_inds)
        else:
            fg_gt_classes = paddle.gather(tgt_labels, fg_inds)

            reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = paddle.gather(deltas, fg_inds)
            reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])
        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True

        if self.bbox_loss is not None:
            reg_delta = self.bbox_transform(reg_delta)
            reg_target = self.bbox_transform(reg_target)
            loss_bbox_reg = self.bbox_loss(
                reg_delta, reg_target).sum() / tgt_labels.shape[0]
            loss_bbox_reg *= self.num_classes
        else:
            loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum(
            ) / tgt_labels.shape[0]

        loss_bbox[reg_name] = loss_bbox_reg

        return loss_bbox

    def bbox_transform(self, deltas, weights=[0.1, 0.1, 0.2, 0.2]):
        wx, wy, ww, wh = weights

        deltas = paddle.reshape(deltas, shape=(0, -1, 4))

        dx = paddle.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        dy = paddle.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        dw = paddle.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        dh = paddle.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dw = paddle.clip(dw, -1.e10, np.log(1000. / 16))
        dh = paddle.clip(dh, -1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = paddle.exp(dw)
        pred_h = paddle.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = paddle.reshape(x1, shape=(-1, ))
        y1 = paddle.reshape(y1, shape=(-1, ))
        x2 = paddle.reshape(x2, shape=(-1, ))
        y2 = paddle.reshape(y2, shape=(-1, ))

        return paddle.concat([x1, y1, x2, y2])

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois
