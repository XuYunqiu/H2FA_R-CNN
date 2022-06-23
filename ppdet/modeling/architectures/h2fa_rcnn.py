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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

from ..layers import GradientScalarLayer
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, KaimingNormal, KaimingUniform
import typing
import math
from ppdet.modeling.initializer import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_

__all__ = ['H2FARCNN']


@register
class H2FARCNN(BaseArch):
    """
    Faster R-CNN network, see https://arxiv.org/abs/1506.01497

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
    """
    __category__ = 'architecture'
    __inject__ = ['bbox_post_process']

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 img_cls_head,
                 local_domain_cls_head,
                 global_domain_cls_head,
                 neck=None):
        super(H2FARCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process

        self.img_cls_head = img_cls_head
        self.local_domain_cls_head = local_domain_cls_head
        self.global_domain_cls_head = global_domain_cls_head

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        # kwargs = {'input_shape': backbone.out_shape}
        kwargs = {'input_shape': [backbone.out_shape[-1]]}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        # out_shape = neck and neck.out_shape or backbone.out_shape
        out_shape = neck and neck.out_shape or [backbone.out_shape[-1]]
        low_out_shape, high_out_shape = backbone.out_shape
        img_cls_head = ImageClsHead(high_out_shape)
        local_domain_cls_head = LocalDAHead(low_out_shape)
        global_domain_cls_head = GlobalDAHead(high_out_shape)

        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
            "img_cls_head": img_cls_head,
            "local_domain_cls_head": local_domain_cls_head,
            "global_domain_cls_head": global_domain_cls_head,
        }

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])

        if self.fuse_norm:
            image = inputs['image']
            self.inputs['image'] = (image * self.scale - self.mean) / self.std
            self.inputs['im_shape'] = inputs['im_shape']
            self.inputs['scale_factor'] = inputs['scale_factor']
        else:
            self.inputs = inputs

        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            inputs_list = []
            # multi-scale input
            if not isinstance(inputs, typing.Sequence):
                inputs_list.append(inputs)
            else:
                inputs_list.extend(inputs)
            outs = []
            for inp in inputs_list:
                if self.fuse_norm:
                    self.inputs['image'] = (
                        inp['image'] * self.scale - self.mean) / self.std
                    self.inputs['im_shape'] = inp['im_shape']
                    self.inputs['scale_factor'] = inp['scale_factor']
                else:
                    self.inputs = inp
                outs.append(self.get_pred())

            # multi-scale test
            if len(outs)>1:
                out = self.merge_multi_scale_predictions(outs)
            else:
                out = outs[0]
        return out

    @paddle.no_grad()
    def _get_image_level_gt(self, inputs):
        """
        Convert instance-level annotations to image-level
        """
        gt_class_img_oh = paddle.zeros(
            (len(inputs['gt_class']), self.bbox_head.num_classes), dtype=paddle.float32
        )

        for i, gt_per_image in enumerate(inputs['gt_class']):
            gt_class_img_oh[i, [paddle.unique(gt_per_image)]] = 1
        gt_class_img_oh.stop_gradient = True

        return gt_class_img_oh


    def _inference(self):
        """forward pipeline for source domain"""
        _, feats = self.backbone(self.inputs)

        if self.neck is not None:
            feats = self.neck([feats])
            feats = feats[0]

        rois, rois_num, _ = self.rpn_head([feats], self.inputs)
        preds, _ = self.bbox_head([feats], rois, rois_num, None)

        im_shape = self.inputs['im_shape']
        scale_factor = self.inputs['scale_factor']
        bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                im_shape, scale_factor)

        # rescale the prediction back to origin image
        bbox_pred = self.bbox_post_process.get_pred(bbox, bbox_num,
                                                    im_shape, scale_factor)
        return bbox_pred, bbox_num

    def _image_level_align(self):
        """forward pipeline for both source and target domain """
        assert self.training, "_image_level_align only used in training"

        low_feats, high_feats = self.backbone(self.inputs)

        img_gt_oh = self._get_image_level_gt(self.inputs)
        ic_loss = self.img_cls_head(high_feats, img_gt_oh)

        local_dc_loss = self.local_domain_cls_head(low_feats)
        global_dc_loss = self.global_domain_cls_head(high_feats)
        dc_loss = {'loss_dc': global_dc_loss['loss_dc']+local_dc_loss['loss_dc']}

        num_target_inputs = img_gt_oh.shape[0] // 2
        target_img_gt_oh = img_gt_oh[num_target_inputs:, ...]

        return ic_loss, dc_loss, high_feats, target_img_gt_oh

    def _instance_level_align(self, high_feats, target_img_gt_oh):
        """forward pipeline for source and target domain"""
        assert self.training, "_instance_level_align only used in training"
        
        num_target_inputs = target_img_gt_oh.shape[0]
        source_inputs = {
            k: self.inputs[k][:num_target_inputs] if type(self.inputs[k]) is not int
            else self.inputs[k]
            for k in self.inputs.keys()
        }
        target_inputs = {
            k: self.inputs[k][num_target_inputs:] if type(self.inputs[k]) is not int
            else self.inputs[k]
            for k in self.inputs.keys()
        }

        s_high_feats, t_high_feats =  high_feats.split(2, axis=0)

        # Instance-level and Image-level Recognition (IIR) unit
        rois, rois_num, rpn_loss = self.rpn_head([s_high_feats], source_inputs)
        bbox_loss, _ = self.bbox_head([s_high_feats], rois, rois_num, source_inputs)

        t_rois, t_rois_num, t_obj_logits = self.rpn_head.forward_weak([t_high_feats], target_inputs)

        img_cls_loss = self.bbox_head.forward_weak(
            [t_high_feats], t_rois, t_rois_num, target_inputs, t_obj_logits, target_img_gt_oh
        )

        return rpn_loss, bbox_loss, img_cls_loss

    def get_loss(self):
        ic_loss, dc_loss, high_feats, target_img_gt_oh = self._image_level_align()
        rpn_loss, bbox_loss, img_cls_loss = self._instance_level_align(high_feats, target_img_gt_oh)

        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        loss.update(img_cls_loss)
        loss.update(ic_loss)
        loss.update(dc_loss)

        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num = self._inference()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output



class GlobalDAHead(nn.Layer):
    """
    Image-level domain discriminator
    """

    # TODO: support different backbones
    def __init__(self, in_channel=2048):
        super(GlobalDAHead, self).__init__()

        self.conv1 = nn.Conv2D(in_channel.channels, 512, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.conv2 = nn.Conv2D(512, 128, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.conv3 = nn.Conv2D(128, 128, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(512)
        self.bn2 = nn.BatchNorm2D(128)
        self.bn3 = nn.BatchNorm2D(128)
        self.fc = nn.Linear(128, 1)
        

        kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        kaiming_uniform_(self.conv2.weight, a=math.sqrt(5))
        kaiming_uniform_(self.conv3.weight, a=math.sqrt(5))
        kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.fc.weight)
        bound = 1 / math.sqrt(fan_in)
        uniform_(self.fc.bias, -bound, bound)

        self.gamma = 5
        grl_weight = 1.0
        self.grl = GradientScalarLayer(-1.0 * grl_weight)

    def forward(self, x):
        x = self.grl(x)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)

        x = F.avg_pool2d(x, (x.shape[2], x.shape[3])).reshape([-1, 128])
        x = self.fc(x)

        da_targets = paddle.zeros_like(x)
        da_targets.stop_gradient = True
        num_source_input = x.shape[0] // 2

        da_targets[:num_source_input, ...] += 1

        losses = F.sigmoid_focal_loss(x, da_targets, gamma=self.gamma, reduction='mean')
        return {'loss_dc': losses}


class LocalDAHead(nn.Layer):
    """
    Local domain classifier
    """

    def __init__(self, in_channel=256):
        super(LocalDAHead, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channel.channels, 256, kernel_size=1, stride=1, padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)), bias_attr=False
        )
        self.conv2 = nn.Conv2D(
            256, 128, kernel_size=1, stride=1, padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)), bias_attr=False
        )
        self.conv3 = nn.Conv2D(
            128, 1, kernel_size=1, stride=1, padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)), bias_attr=False
        )

        grl_weight = 1.0
        self.grl = GradientScalarLayer(-1.0 * grl_weight)

    def forward(self, x):
        x = self.grl(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))

        da_targets = paddle.zeros_like(x)
        da_targets.stop_gradient = True
        num_source_input = x.shape[0] // 2
        da_targets[:num_source_input, ...] += 1

        losses = F.mse_loss(x, da_targets, reduction='mean')

        return {'loss_dc': losses}


class ImageClsHead(nn.Layer):
    """
    Classification head for image-level multi-label classification
        and image-level class-specific feature alignment
    """

    def __init__(self, in_channel, num_classes=20):
        super(ImageClsHead, self).__init__()
        self.num_classes = num_classes

        self.conv = nn.Conv2D(
            in_channel.channels, self.num_classes, kernel_size=1, bias_attr=False
        )
        from ppdet.modeling.initializer import kaiming_normal_, constant_
        kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            constant_(self.conv.bias, value=0.)

    def forward(self, x, gt_classes_img_oh):
        x = self.conv(x)
        logits = F.avg_pool2d(x, (x.shape[2], x.shape[3]))
        logits = logits.reshape([-1, self.num_classes])

        if gt_classes_img_oh.sum() == 0:
            return {'loss_ic': 0.0 * logits.sum()}

        losses = F.binary_cross_entropy_with_logits(
            logits, gt_classes_img_oh, reduction='mean'
        )
        return {'loss_ic': losses*0.1}


