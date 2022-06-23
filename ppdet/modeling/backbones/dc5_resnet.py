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


import math
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec

from .resnet import Blocks, BasicBlock, BottleNeck, ConvNormLayer

__all__ = ['DC5ResNet', 'DC5BottleNeck']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class DilatedConvNormLayer(nn.Layer):
    def __init__(
            self,
            ch_in,
            ch_out,
            filter_size,
            stride,
            groups=1,
            act=None,
            norm_type='bn',
            norm_decay=0.,
            freeze_norm=True,
            lr=1.0,
            dilation=2,
    ):
        super(DilatedConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size= filter_size, # 3
            stride=stride, # 1
            padding=1 * dilation, # 2
            dilation=dilation, # 2
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr),
            bias_attr=False)

        norm_lr = 0. if freeze_norm else lr
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)

        global_stats = True if freeze_norm else None
        if norm_type in ['sync_bn', 'bn']:
            self.norm = nn.BatchNorm2D(
                ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr,
                use_global_stats=global_stats)
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        out = self.conv(inputs)

        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out


class DC5BottleNeck(nn.Layer):

    expansion = 4

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=4,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(DC5BottleNeck, self).__init__()
        """
        The original MSRA ResNet models have stride in the first 1x1 conv 
        The subsequent fb.torch.resnet, Caffe2 ResNe[X]t and paddle.resnet implementations have stride in the 3x3 conv 
        The D2 uses MSRA pre-train (variant a), while ppdet uses paddle pre-train (variant b)
        """
        # TODO: try `variant a' as `MODEL.RESNETS.STRIDE_IN_1X1=True' set in D2
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride # stride1=1, stride2=2

        # ResNeXt
        width = int(ch_out * (base_width / 64.)) * groups

        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvNormLayer(
                ch_in=ch_in,
                ch_out=ch_out * self.expansion,
                filter_size=1,
                # stride=stride,
                stride=1,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width,
            filter_size=1,
            stride=stride1,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = DilatedConvNormLayer(
            ch_in=width,
            ch_out=width,
            filter_size=3,
            # stride=stride2,
            stride=1,
            groups=groups,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dilation=2,)

        self.branch2c = ConvNormLayer(
            ch_in=width,
            ch_out=ch_out * self.expansion,
            filter_size=1,
            stride=1,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

    def forward(self, inputs):

        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)

        return out


@register
@serializable
class DC5ResNet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 ch_in=64,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 num_stages=4,
                 std_senet=False):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385
        
        Args:
            depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
            ch_in (int): output channel of first stage, default 64
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                                 lower learning rate ratio is need for pretrained model 
                                 got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            freeze_norm (bool): freeze normalization layers
            freeze_at (int): freeze the backbone at which stage
            return_idx (list): index of the stages whose feature maps are returned
            dcn_v2_stages (list): index of stages who select deformable conv v2
            num_stages (int): total num of stages
            std_senet (bool): whether use senet, default True
        """
        super(DC5ResNet, self).__init__()
        self._model_type = 'ResNet' if groups == 1 else 'ResNeXt'
        assert num_stages >= 1 and num_stages <= 4
        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)

        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_sublayer(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0))

        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        dc_block = DC5BottleNeck

        self._out_channels = [block.expansion * v for v in ch_out_list]
        # self._out_strides = [4, 8, 16, 32]
        self._out_strides = [4, 8, 16, 16] # ResNet DC5

        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            if i == num_stages-1:
                res_layer = self.add_sublayer(
                    res_name,
                    Blocks(
                        dc_block,
                        self.ch_in,
                        ch_out_list[i],
                        count=block_nums[i],
                        name_adapter=na,
                        stage_num=stage_num,
                        variant=variant,
                        groups=groups,
                        base_width=base_width,
                        lr=lr_mult,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        dcn_v2=(i in self.dcn_v2_stages),
                        std_senet=std_senet))
            else:
                res_layer = self.add_sublayer(
                    res_name,
                    Blocks(
                        block,
                        self.ch_in,
                        ch_out_list[i],
                        count=block_nums[i],
                        name_adapter=na,
                        stage_num=stage_num,
                        variant=variant,
                        groups=groups,
                        base_width=base_width,
                        lr=lr_mult,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        dcn_v2=(i in self.dcn_v2_stages),
                        std_senet=std_senet))
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at + 1, num_stages)):
                self._freeze_parameters(self.res_layers[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.stop_gradient = True

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
