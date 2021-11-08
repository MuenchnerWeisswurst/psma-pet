from typing import List, Tuple

from src.models.blocks import ResNet3DBlock, DenseBlock3D, TransitionBlock3D, TransitionBlock2D, DenseBlock2D
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet3D(nn.Module):
    """
    ResNet for 3D input
    """
    def __init__(self,
                 n_blocks: int,
                 residuals_per_block: List[int],
                 channels_per_block: List[Tuple[int, int]],
                 batchnorm: bool = True,
                 dropout: List[float] = None,
                 activation=nn.ReLU,
                 in_channels: int = 1,
                 ):
        """

        :param n_blocks: Number of ResNet blocks
        :param residuals_per_block: List of number of residuals per block
        :param channels_per_block: List of tuple of input and output channels per block
        :param batchnorm: Batchnorm
        :param dropout: Dropout probability
        :param activation: Activation function
        :param in_channels: Number of input channels for the first layer
        """
        super(ResNet3D, self).__init__()
        if not dropout:
            dropout = n_blocks * [None]
        elif isinstance(dropout, float):
            dropout = n_blocks * [dropout]
        else:
            dropout = dropout

        init_layer = [
            nn.Conv3d(in_channels, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.BatchNorm3d(64),
            activation(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        ]
        self.net = nn.Sequential(
            *init_layer,
            *[
                ResNet3DBlock(residuals_per_block[i], channels_per_block[i][0], channels_per_block[i][1],
                              batchnorm=batchnorm, dropout=dropout[i], activation=activation)
                for i in range(n_blocks)
            ],
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(channels_per_block[-1][-1], 1)
        )

    def forward(self, X):
        return self.net(X)


class DenseNet3D(nn.Module):
    """
    DenseNet for 3D input
    """
    def __init__(self,
                 num_channels: int,
                 growth_rate: int,
                 channels_per_block: List[int],
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 glob_pooling=nn.AdaptiveMaxPool3d,
                 in_channels: int = 1,
                 ):
        """

        :param num_channels: Number of channels after first layer
        :param growth_rate: Growth rate
        :param channels_per_block: List of number of ConvLayer per DenseBlock
        :param activation: Activation activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        :param glob_pooling: Final pooling layer
        :param in_channels: Number of channels for first layer
        """
        super(DenseNet3D, self).__init__()
        init_layer = [
            nn.Conv3d(in_channels, num_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.BatchNorm3d(num_channels),
            activation(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        ]
        layers = []
        for i, num_convs in enumerate(channels_per_block):
            layers.append(DenseBlock3D(num_convs, num_channels, growth_rate, activation, kernel_size, padding))
            num_channels += num_convs * growth_rate
            if i != len(channels_per_block) - 1:
                layers.append(TransitionBlock3D(num_channels, num_channels // 2))
                num_channels //= 2

        self.net = nn.Sequential(
            *init_layer,
            *layers,
            nn.BatchNorm3d(num_channels),
            activation(),
            glob_pooling((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


class DenseNet2D(nn.Module):
    """
    DenseNet for 2D input
    """
    def __init__(self,
                 num_channels: int,
                 growth_rate: int,
                 channels_per_block: List[int],
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3),
                 padding: Tuple[int, int, int] = (1, 1),
                 glob_pooling=nn.AdaptiveMaxPool2d,
                 in_channels: int = 1,
                 ):
        """

        :param num_channels: Number of channels after first layer
        :param growth_rate: Growth rate
        :param channels_per_block: List of number of ConvLayer per DenseBlock
        :param activation: Activation activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        :param glob_pooling: Final pooling layer
        :param in_channels: Number of channels for first layer
        """
        super(DenseNet2D, self).__init__()
        init_layer = [
            nn.Conv2d(in_channels, num_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(num_channels),
            activation(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ]
        layers = []
        for i, num_convs in enumerate(channels_per_block):
            layers.append(DenseBlock2D(num_convs, num_channels, growth_rate, activation, kernel_size, padding))
            num_channels += num_convs * growth_rate
            if i != len(channels_per_block) - 1:
                layers.append(TransitionBlock2D(num_channels, num_channels // 2, activation=activation))
                num_channels //= 2

        self.net = nn.Sequential(
            *init_layer,
            *layers,
            nn.BatchNorm2d(num_channels),
            activation(),
            glob_pooling((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


class DenseNet3DWithScalars(nn.Module):
    """
    DenseNet for 3D and scalar input
    """
    def __init__(self,
                 num_channels: int,
                 growth_rate: int,
                 channels_per_block: List[int],
                 scalar_hidden_dims: List[int],
                 classifier_hidden_dims: List[int],
                 activation=nn.ReLU,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 glob_pooling=nn.AdaptiveMaxPool3d,
                 in_channels: int = 1,
                 ):
        """

        :param num_channels: Number of channels after first layer
        :param growth_rate: Growth rate
        :param channels_per_block: List of number of ConvLayer per DenseBlock
        :param scalar_hidden_dims: List of number of neurons for hidden scalar net
        :param classifier_hidden_dims: List of number of neurons for hidden classification net
        :param activation: Activation activation function
        :param kernel_size: Kernel size
        :param padding: Padding
        :param glob_pooling: Final pooling layer
        :param in_channels: Number of channels for first layer
        """
        super(DenseNet3DWithScalars, self).__init__()
        init_layer = [
            nn.Conv3d(in_channels, num_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.BatchNorm3d(num_channels),
            activation(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        ]
        layers = []
        for i, num_convs in enumerate(channels_per_block):
            layers.append(DenseBlock3D(num_convs, num_channels, growth_rate, activation, kernel_size, padding))
            num_channels += num_convs * growth_rate
            if i != len(channels_per_block) - 1:
                layers.append(TransitionBlock3D(num_channels, num_channels // 2))
                num_channels //= 2

        self.dense = nn.Sequential(
            *init_layer,
            *layers,
            nn.BatchNorm3d(num_channels),
            activation(),
            glob_pooling((1, 1, 1)),
            nn.Flatten(),
        )
        layers = [nn.Linear(1, scalar_hidden_dims[0]), nn.BatchNorm1d(scalar_hidden_dims[0]), activation()]
        for i in range(1, len(scalar_hidden_dims)):
            layers.append(nn.Linear(scalar_hidden_dims[i - 1], scalar_hidden_dims[i]))
            layers.append(nn.BatchNorm1d(scalar_hidden_dims[i]))
            layers.append(activation())
        self.scalar_layer = nn.Sequential(
            *layers
        )

        if len(classifier_hidden_dims) > 0:
            layers = [nn.Linear(num_channels + scalar_hidden_dims[-1], classifier_hidden_dims[0])]
            for i in range(1, len(classifier_hidden_dims)):
                layers.append(nn.Linear(classifier_hidden_dims[i - 1], classifier_hidden_dims[i]))
                layers.append(nn.BatchNorm1d(classifier_hidden_dims[i]))
                layers.append(activation())
            layers.append(nn.Linear(classifier_hidden_dims[-1], 1))
        else:
            layers = [nn.Linear(num_channels + scalar_hidden_dims[-1], 1)]

        self.classifier = nn.Sequential(
            *layers
        )

    def forward(self, x):
        pre_psa, image = x['psa'], x['image']
        hidden_pre_psa = self.scalar_layer(pre_psa)
        hidden_image = self.dense(image)
        pred = self.classifier(torch.cat([hidden_image, hidden_pre_psa], dim=-1))
        return pred
