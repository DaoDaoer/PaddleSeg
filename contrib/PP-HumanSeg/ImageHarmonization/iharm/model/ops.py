import paddle
from paddle import nn as nn


class SimpleInputFusion(nn.Layer):
    def __init__(self, add_ch=1, rgb_ch=3, ch=8, norm_layer=nn.BatchNorm2D):
        super(SimpleInputFusion, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=add_ch + rgb_ch, out_channels=ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            norm_layer(ch),
            nn.Conv2d(in_channels=ch, out_channels=rgb_ch, kernel_size=1),
        )

    def forward(self, image, additional_input):
        return self.fusion_conv(
            paddle.concat((image, additional_input), axis=1))


class ChannelAttention(nn.Layer):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_pools = nn.LayerList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        ])
        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(
                len(self.global_pools) * in_channels,
                intermediate_channels_count),
            nn.ReLU(),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled_x = []
        for global_pool in self.global_pools:
            pooled_x.append(global_pool(x))
        pooled_x = paddle.concat(pooled_x, axis=1).flatten(start_axis=1)
        channel_attention_weights = self.attention_transform(
            pooled_x)[..., None, None]
        return channel_attention_weights * x


class MaskedChannelAttention(nn.Layer):
    def __init__(self, in_channels, *args, **kwargs):
        super(MaskedChannelAttention, self).__init__()
        self.global_max_pool = MaskedGlobalMaxPool2d()
        self.global_avg_pool = FastGlobalAvgPool2d()

        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(3 * in_channels, intermediate_channels_count),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        if mask.shape[2:] != x.shape[:2]:
            mask = nn.functional.interpolate(
                mask, size=x.size()[-2:], mode='bilinear', align_corners=True)
        pooled_x = paddle.concat(
            [self.global_max_pool(x, mask),
             self.global_avg_pool(x)], axis=1)
        channel_attention_weights = self.attention_transform(
            pooled_x)[..., None, None]

        return channel_attention_weights * x


class MaskedGlobalMaxPool2d(nn.Layer):
    def __init__(self):
        super().__init__()
        self.global_max_pool = FastGlobalMaxPool2d()

    def forward(self, x, mask):
        return paddle.concat((self.global_max_pool(x * mask),
                              self.global_max_pool(x * (1.0 - mask))),
                             axis=1)


class FastGlobalAvgPool2d(nn.Layer):
    def __init__(self):
        super(FastGlobalAvgPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.reshape((in_size[0], in_size[1], -1)).mean(axis=2)


class FastGlobalMaxPool2d(nn.Layer):
    def __init__(self):
        super(FastGlobalMaxPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.reshape((in_size[0], in_size[1], -1)).max(axis=2)[0]


class ScaleLayer(nn.Layer):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            paddle.full((1, ), init_value / lr_mult, dtype=paddle.float32))

    def forward(self, x):
        scale = paddle.abs(self.scale * self.lr_mult)
        return x * scale


class FeaturesConnector(nn.Layer):
    def __init__(self, mode, in_channels, feature_channels, out_channels):
        super(FeaturesConnector, self).__init__()
        self.mode = mode if feature_channels else ''

        if self.mode == 'catc':
            self.reduce_conv = nn.Conv2d(
                in_channels + feature_channels, out_channels, kernel_size=1)
        elif self.mode == 'sum':
            self.reduce_conv = nn.Conv2d(
                feature_channels, out_channels, kernel_size=1)

        self.output_channels = out_channels if self.mode != 'cat' else in_channels + feature_channels

    def forward(self, x, features):
        if self.mode == 'cat':
            return paddle.concat((x, features), 1)
        if self.mode == 'catc':
            return self.reduce_conv(paddle.concat((x, features), 1))
        if self.mode == 'sum':
            return self.reduce_conv(features) + x
        return x

    def extra_repr(self):
        return self.mode
