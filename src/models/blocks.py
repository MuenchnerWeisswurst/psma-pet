from typing import Tuple

import torch
import torch.nn as nn


class Residual3DBlock(nn.Module):
    """
    Resdiual block with pre-activation (ResNetV2),
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 batchnorm: bool = True,
                 dropout: float = None,
                 conv1x1: bool = False,
                 activation=nn.ReLU,
                 ):
        """

        :param in_channels: Num input channels
        :param out_channels: Num out channels
        :param kernel: Kernel size
        :param stride: Stride
        :param batchnorm: Batchnorm
        :param dropout: Dropout probability
        :param conv1x1: Whether residual path contains a conv1x1 layer
        :param activation: Activation function
        """
        super(Residual3DBlock, self).__init__()
        self.bn1 = nn.Identity()
        self.activation1 = activation()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=(1, 1, 1))

        self.bn2 = nn.Identity()
        self.activation2 = activation()
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, padding=(1, 1, 1))

        if batchnorm:
            self.bn1 = nn.BatchNorm3d(in_channels)
            self.bn2 = nn.BatchNorm3d(in_channels)

        self.dropout = nn.Identity()
        if dropout:
            self.dropout = nn.Dropout3d(dropout)

        self.conv3 = nn.Identity()
        if conv1x1:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride)

    def forward(self, X):
        X_ = self.bn1(X)
        X_ = self.activation1(X_)
        X_ = self.conv1(X_)

        X_ = self.bn2(X_)
        X_ = self.activation2(X_)
        X_ = self.conv2(X_)

        X_ = self.dropout(X_)

        X = self.conv3(X)

        X_ = X + X_

        return X_


class ResNet3DBlock(nn.Module):
    """
    ResNet block
    """
    def __init__(self,
                 n_residuals: int,
                 in_channels: int,
                 out_channels: int,
                 batchnorm: bool = True,
                 dropout: float = None,
                 activation=nn.ReLU,
                 ):
        """

        :param n_residuals: Num of residual blocks
        :param in_channels: Num of input channels
        :param out_channels: Num of output channels
        :param batchnorm: Batchnorm
        :param dropout: Dropout probability
        :param activation: Activation function
        """
        super(ResNet3DBlock, self).__init__()
        self.block = nn.Sequential(*[
            Residual3DBlock(in_channels, out_channels, stride=(2, 2, 2), conv1x1=True, activation=activation,
                            batchnorm=batchnorm, dropout=dropout)
            if i == 0 else Residual3DBlock(out_channels, out_channels, activation=activation, batchnorm=batchnorm,
                                           dropout=dropout)
            for i in range(n_residuals)
        ])

    def forward(self, X):
        return self.block(X)


class DenseConv3DBlock(nn.Module):
    """
    ConvLayer of a DenseBlock3D
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 padding: Tuple[int, int, int] = (1, 1, 1)
                 ):
        """

        :param in_channels: Num input channels
        :param out_channels: Num output channels
        :param activation: Activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        """
        super(DenseConv3DBlock, self).__init__()

        self.bn = nn.BatchNorm3d(in_channels)
        self.activation = activation()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(self.activation(self.bn(x)))


class DenseConv2DBlock(nn.Module):
    """
    ConvLayer of a DenseBlock2D
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 padding: Tuple[int, int, int] = (1, 1, 1)
                 ):
        """

        :param in_channels: Num input channels
        :param out_channels: Num output channels
        :param activation: Activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        """
        super(DenseConv2DBlock, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(self.activation(self.bn(x)))


class DenseBlock3D(nn.Module):
    """
    DenseBlock
    """
    def __init__(self, num_conv: int,
                 in_channels: int,
                 num_channels: int,
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3),
                 padding: Tuple[int, int, int] = (1, 1)
                 ):
        """

        :param num_conv: Num of ConvLayer
        :param in_channels: Num of input channels
        :param num_channels: Num of output channels for each ConvLayer
        :param activation: Activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        """
        super(DenseBlock3D, self).__init__()

        self.block = nn.Sequential(*[
            DenseConv3DBlock(num_channels * i + in_channels, num_channels, activation, kernel_size, padding)
            for i in range(num_conv)
        ])

    def forward(self, X):
        for layer in self.block:
            X_ = layer(X)
            X = torch.cat((X, X_), dim=1)

        return X


class DenseBlock2D(nn.Module):
    """
    DenseBlock
    """
    def __init__(self, num_conv: int,
                 in_channels: int,
                 num_channels: int,
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3),
                 padding: Tuple[int, int, int] = (1, 1)
                 ):
        """

        :param num_conv: Num of ConvLayer
        :param in_channels: Num of input channels
        :param num_channels: Num of output channels for each ConvLayer
        :param activation: Activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        """
        super(DenseBlock2D, self).__init__()

        self.block = nn.Sequential(*[
            DenseConv2DBlock(num_channels * i + in_channels, num_channels, activation, kernel_size, padding)
            for i in range(num_conv)
        ])

    def forward(self, X):
        for layer in self.block:
            X_ = layer(X)
            X = torch.cat((X, X_), dim=1)

        return X


class TransitionBlock3D(nn.Module):
    """
    TransitionBlock
    """
    def __init__(self,
                 in_channels: int,
                 num_channels: int,
                 activation=nn.ReLU,
                 ):
        """

        :param in_channels: Num of input channels
        :param num_channels: Num of output channels
        :param activation: Activation function
        """
        super(TransitionBlock3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.activation = activation()
        self.conv = nn.Conv3d(in_channels, num_channels, kernel_size=(1, 1, 1))
        self.avg_pool = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2))

    def forward(self, x):
        return self.avg_pool(self.conv(self.activation(self.bn(x))))


class TransitionBlock2D(nn.Module):
    """
    TransitionBlock
    """
    def __init__(self,
                 in_channels: int,
                 num_channels: int,
                 activation=nn.ReLU,
                 ):
        """

        :param in_channels: Num of input channels
        :param num_channels: Num of output channels
        :param activation: Activation function
        """
        super(TransitionBlock2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation()
        self.conv = nn.Conv2d(in_channels, num_channels, kernel_size=(1, 1))
        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        return self.avg_pool(self.conv(self.activation(self.bn(x))))
