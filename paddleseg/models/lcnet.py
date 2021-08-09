# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal

from paddleseg.models.fcn import FCNHead
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init

# from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
# from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


def load_dygraph_pretrain(**kwargs):
    return


def load_dygraph_pretrain_from_url(**kwargs):
    return


MODEL_URLS = {
    "LCNet_x0_125":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x0_125_pretrained.pdparams",
    "LCNet_x0_175":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x0_175_pretrained.pdparams",
    "LCNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x0_25_pretrained.pdparams",
    "LCNet_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x0_35_pretrained.pdparams",
    "LCNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x0_5_pretrained.pdparams",
    "LCNet_x0_7":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x0_7_pretrained.pdparams",
    "LCNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/LCNet_x1_0_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False], [3, 32, 64, 2, False]],
    "blocks3": [[3, 64, 64, 1, False], [3, 64, 128, 2, False]],
    "blocks4": [[3, 128, 128, 1, False], [3, 128, 256, 2, False]],
    "blocks5": [[5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 512, 2, True]],
    "blocks6": [[5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


@manager.MODELS.add_component
class LC_Seg(nn.Layer):
    def __init__(self,
                 class_num,
                 scale=1.0,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.scale = scale
        self.backbone = LC_Backbone(scale)
        self.head = FCNHead(class_num, backbone_channels=(512, ))
        self.align_corners = align_corners
        # self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)


class LC_Backbone(nn.Layer):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks2(x)
        print(x.shape)
        x = self.blocks3(x)
        print(x.shape)
        x = self.blocks4(x)
        print(x.shape)
        x = self.blocks5(x)
        print(x.shape)
        x = self.blocks6(x)
        print(x.shape)

        return [x]


class LCNet(nn.Layer):
    def __init__(self,
                 scale=1.0,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.avg_pool = AdaptiveAvgPool2D(1)

        self.last_conv = Conv2D(
            in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)

        self.hardswish = nn.Hardswish()
        self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)

        self.fc = Linear(self.class_expand, class_num)

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks2(x)
        print(x.shape)
        x = self.blocks3(x)
        print(x.shape)
        x = self.blocks4(x)
        print(x.shape)
        x = self.blocks5(x)
        print(x.shape)
        x = self.blocks6(x)
        print(x.shape)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def LCNet_x0_125(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x0_125
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x0_125` model depends on args.
    """
    model = LCNet(scale=0.125, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x0_125"], use_ssld)
    return model


def LCNet_x0_175(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x0_175
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x0_175` model depends on args.
    """
    model = LCNet(scale=0.175, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x0_175"], use_ssld)
    return model


def LCNet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x0_25` model depends on args.
    """
    model = LCNet(scale=0.25, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x0_25"], use_ssld)
    return model


def LCNet_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x0_35` model depends on args.
    """
    model = LCNet(scale=0.35, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x0_35"], use_ssld)
    return model


def LCNet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x1_0` model depends on args.
    """
    model = LCNet(scale=0.5, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x0_5"], use_ssld)
    return model


def LCNet_x0_7(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x0_7
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x0_7` model depends on args.
    """
    model = LCNet(scale=0.7, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x0_7"], use_ssld)
    return model


def LCNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    LCNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `LCNet_x1_0` model depends on args.
    """
    model = LC_Seg(scale=1.0, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["LCNet_x1_0"], use_ssld)
    return model


if __name__ == "__main__":
    model = LCNet_x1_0(class_num=19)
    # old_state_dict = paddle.load("CPULiteNet_weight/CPULiteNet_x0_5/epoch_358.pdparams")
    # state_dict = model.state_dict()
    # for (k_old, v_old), (k_new, v_new) in zip(old_state_dict.items(), state_dict.items()):
    #     state_dict[k_new] = v_old
    x = paddle.randn([1, 3, 1024, 1024])
    y = model(x)
    print(y[0].shape)
    # paddle.save(state_dict, "LCNet_x1_0.pdparams")
    # FLOPs = paddle.flops(model, [1, 3, 224, 224], print_detail=True)

    # model = LCNet_x1_0()
