# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by Yunqiu Xu for H2FA R-CNN
# ------------------------------------------------------------------------

import logging
import numpy as np
from typing import Optional, Tuple, Dict
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers import ShapeSpec, GradientScalarLayer

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            global_da_heads: nn.Module,
            local_da_heads: nn.Module,
            cam_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.global_da_heads = global_da_heads
        self.local_da_heads = local_da_heads
        self.cam_heads = cam_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "global_da_heads": GlobalDAHead(backbone.output_shape()),
            "local_da_heads": LocalDAHead(backbone.output_shape()),
            "cam_heads": CAMHead(backbone.output_shape(), cfg.MODEL.ROI_HEADS.NUM_CLASSES),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # batch_inputs = [target_inputs, source_inputs]
        # source inputs
        num_source_input = len(images) // 2
        source_images = ImageList(
            images.tensor[num_source_input:], images.image_sizes[num_source_input:]
        )
        source_gt_instances = gt_instances[num_source_input:]
        source_features = {}
        for feat_name in self.backbone._out_features:
            source_features[feat_name] = features[feat_name][num_source_input:]
        # target inputs
        target_images = ImageList(
            images.tensor[:num_source_input], images.image_sizes[:num_source_input]
        )
        target_gt_instances = gt_instances[:num_source_input]
        target_features = {}
        for feat_name in self.backbone._out_features:
            target_features[feat_name] = features[feat_name][:num_source_input]

        # (1) image-level class-agnostic alignment
        global_dc_losses = self.global_da_heads(features[self.roi_heads.in_features[0]])  # res5 or res4
        local_dc_losses = self.local_da_heads(features[list(features.keys())[0]])  # res2

        # (2) image-level class-wise alignment
        ic_losses = self.cam_heads(features[self.roi_heads.in_features[0]], gt_instances)

        # Instance-level and Image-level recognition (IIR) unit:
        # (3) instance-level foreground alignment
        if self.proposal_generator:
            source_proposals, source_proposal_losses = self.proposal_generator(
                source_images, source_features, source_gt_instances
            )

            target_proposals = self.proposal_generator.forward_weak_w_grad(
                target_images, target_features
            )

            proposals = target_proposals + source_proposals
            proposal_losses = source_proposal_losses
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # (4) instance-level class-wise alignment
        _, detector_losses = self.roi_heads(
            source_images, source_features, source_proposals, source_gt_instances
        )
        img_cls_losses = self.roi_heads.forward_weak(
            target_features, target_proposals, target_gt_instances
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(img_cls_losses)
        losses.update(proposal_losses)
        losses.update(global_dc_losses)
        losses.update(local_dc_losses)
        losses.update(ic_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results


class GlobalDAHead(nn.Module):
    """
    Global domain classifier for image-level class-agnostic alignment
    """

    def __init__(self, backbone_out_shape: Dict[str, ShapeSpec]):
        super(GlobalDAHead, self).__init__()
        if 'res5' in backbone_out_shape.keys():
            in_channels = 2048
        elif 'res4' in backbone_out_shape.keys():
            in_channels = 1024
        elif 'plain5' in backbone_out_shape.keys():
            in_channels = 512
        else:
            raise KeyError("Unknown backbone output name: {}".format(backbone_out_shape.keys()))

        self.da_conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.da_conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.da_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.da_bn1 = nn.BatchNorm2d(512)
        self.da_bn2 = nn.BatchNorm2d(128)
        self.da_bn3 = nn.BatchNorm2d(128)
        self.da_fc = nn.Linear(128, 1)

        self.gama = 5
        grl_weight = 1.0
        self.grl = GradientScalarLayer(-1.0 * grl_weight)

    def forward(self, x):
        x = self.grl(x)

        x = F.dropout(F.relu(self.da_bn1(self.da_conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.da_bn2(self.da_conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.da_bn3(self.da_conv3(x))), training=self.training)

        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        x = self.da_fc(x)

        da_targets = torch.zeros_like(x, requires_grad=False)
        num_source_input = x.shape[0] // 2
        da_targets[:num_source_input, ...] += 1
        losses = sigmoid_focal_loss_jit(x, da_targets, gamma=self.gama, reduction='mean')

        return {'loss_global_da': losses}


class LocalDAHead(nn.Module):
    """
    Local domain classifier for image-level class-agnostic feature alignment
    """

    def __init__(self, backbone_out_shape: Dict[str, ShapeSpec]):
        super(LocalDAHead, self).__init__()
        if 'res2' in backbone_out_shape.keys():
            in_channels = 256
        elif 'plain2' in backbone_out_shape.keys():
            in_channels = 128
        else:
            print(backbone_out_shape.keys())
            raise KeyError("Unknown backbone output name: {}".format(backbone_out_shape.keys()))

        self.da_conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.da_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.da_conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self._init_weights()

        grl_weight = 1.0
        self.grl = GradientScalarLayer(-1.0 * grl_weight)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)

        normal_init(self.da_conv1, 0, 0.01)
        normal_init(self.da_conv2, 0, 0.01)
        normal_init(self.da_conv3, 0, 0.01)

    def forward(self, x):
        x = self.grl(x)

        x = F.relu(self.da_conv1(x))
        x = F.relu(self.da_conv2(x))
        x = torch.sigmoid(self.da_conv3(x))

        da_targets = torch.zeros_like(x, requires_grad=False)
        num_source_input = x.shape[0] // 2
        da_targets[:num_source_input, ...] += 1
        losses = F.mse_loss(x, da_targets, reduction='mean')

        return {'loss_local_da': losses}


class CAMHead(nn.Module):
    """
    Image-level multi-label classifier for image-level class-wise alignment
    """

    def __init__(self, backbone_out_shape: Dict[str, ShapeSpec], num_classes: int):
        super(CAMHead, self).__init__()
        if 'res5' in backbone_out_shape.keys():
            in_channels = 2048
        elif 'res4' in backbone_out_shape.keys():
            in_channels = 1024
        elif 'plain5' in backbone_out_shape.keys():
            in_channels = 512
        else:
            raise KeyError("Unknown backbone output name: {}".format(backbone_out_shape.keys()))
        self.num_classes = num_classes

        self.cam_conv = nn.Conv2d(in_channels, self.num_classes, kernel_size=1, bias=False)
        weight_init.c2_msra_fill(self.cam_conv)

    def forward(self, x, gt_instances):
        x = self.cam_conv(x)

        logits = F.avg_pool2d(x, (x.size(2), x.size(3)))
        logits = logits.view(-1, self.num_classes)

        if gt_instances is None:
            return {'loss_cam': 0.0 * logits.sum()}

        gt_classes_img_oh = self.get_image_level_gt(gt_instances)

        losses = F.binary_cross_entropy_with_logits(
            logits, gt_classes_img_oh, reduction='mean'
        )
        return {'loss_cam': losses * 0.1}

    @torch.no_grad()
    def get_image_level_gt(self, targets):
        """
        Convert instance-level annotations to image-level
        """
        gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
        gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
        gt_classes_img_oh = torch.cat(
            [
                torch.zeros(
                    (1, self.num_classes), dtype=torch.float, device=gt_classes_img[0].device
                ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
                for gt in gt_classes_img_int
            ],
            dim=0,
        )
        return gt_classes_img_oh