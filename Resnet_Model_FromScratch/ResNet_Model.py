import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


# Create an Autopadding ConV2D
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dAuto, self).__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3X3 = partial(Conv2dAuto, kernel_size=3, bias=False)


# conv = conv3X3(in_channels=32, out_channels=64)
# print(conv)

def activation_func(activation):
    return nn.ModuleDict(dict(relu=nn.ReLU(inplace=True), leaky_relu=nn.LeakyReLU(negative_slope=0.01, inplace=True),
                              selu=nn.SELU(inplace=True), none=nn.Identity()))[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_func(activation)

        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, X):
        residual = X
        if self.apply_shortcut:
            residual = self.shortcut(X)
        X = self.blocks(X)
        X += residual
        X = self.activation(X)
        return X

    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels


# res = ResidualBlock(32, 64)
# print(res)
# dummy = torch.ones((1,1,1,1))
# block = ResidualBlock(1, 64)
# print(block(dummy))

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1,
                 downsampling=1, conv=conv3X3, *args, **kwargs):
        super(ResNetResidualBlock, self).__init__(in_channels, out_channels, *args, **kwargs)

        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels))


class ResnetBasicBlock(ResNetResidualBlock):
    """
    Basic Resnet Block of two layers 3X3 conv/BN/Activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(ResnetBasicBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            self.activation,
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResnetBottleNeckBlock(ResNetResidualBlock):
    """
    Basic Resnet Block of two layers 3X3 conv/BN/Activation
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(ResnetBottleNeckBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, kernel_size=1),
            self.activation,
            conv_bn(self.out_channels, self.out_channels, conv=self.conv, bias=False, kernel_size=3,
                    stride=self.downsampling),
            self.activation,
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, kernel_size=1),
        )


# dummy = torch.ones((1, 32, 10, 10))
# block = ResnetBottleNeckBlock(32, 64)
# block(dummy).shape
# print(block)


class ResNetLayer(nn.Module):
    """
    n layers of Resnet blocks one after the other
    """

    def __init__(self, in_channels, out_channels, block=ResnetBasicBlock, n=1, *args, **kwargs):
        super(ResNetLayer, self).__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels,
                    downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        return self.blocks(x)


# dummy = torch.ones((1, 64, 48, 48))
# layer = ResNetLayer(64, 128, block=ResnetBasicBlock, n=3)
# print(layer(dummy).shape) # ==> torch.Size([1, 128, 24, 24])

class ResNetEncoder(nn.Module):

    def __init__(self, in_channels=3, block_sizes=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2], activation='relu', block=ResnetBasicBlock, *args, **kwargs):
        super(ResNetEncoder, self).__init__()
        self.block_sizes = block_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_out_block_size = list(zip(block_sizes, block_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(block_sizes[0], block_sizes[0], n=depths[0], activation=activation, block=block, *args, **kwargs),
            *[ResNetLayer(in_channels*block.expansion,
                          out_channels, n=n, activation=activation, block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_size, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNetDecoder(nn.Module):

    def __init__(self, in_features, n_classes):
        super(ResNetDecoder, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = nn.Flatten()(x)  # x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super(ResNet, self).__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet18(in_channels, n_classes, block=ResnetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[2, 2, 2, 2], *args, **kwargs)

def resnet50(in_channels, n_classes, block=ResnetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[3, 4, 6, 3], *args, **kwargs)

"""
For 18: [2, 2, 2, 2] ResnetBasicBlock
For 50: [3, 4, 6, 3] ResnetBottleNeckBlock
for 101: [3, 4, 23, 3] ResnetBottleNeckBlock
for 152: [3, 8, 36, 3] ResnetBottleNeckBlock
"""



if __name__ == '__main__':
    model = resnet18(3, 2)
    print(model)