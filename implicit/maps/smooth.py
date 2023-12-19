"""
Simple, smooth maps R^n -> R^m.
"""
import math
from abc import abstractmethod

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from labml_nn.unet import UNet, DoubleConvolution, DownSample, UpSample, CropAndConcat

from .base import Map


class SmoothMap(Map):
    """Base class for simple smooth maps."""

    def _listify_hypers(self, num_layers, hyperparams: dict):
        for name, value in hyperparams.items():
            if isinstance(value, int) or value is None:
                self.__setattr__(name, [value] * num_layers)
            else:
                assert len(value) == num_layers
                self.__setattr__(name, value)

    def _apply_spectral_norm(self):
        for module in self.modules():
            if "weight" in module._parameters:
                nn.utils.spectral_norm(module)

    def forward(self, x):
        return self.net(x)


class FlatSmoothMap(SmoothMap):
    """Smooth map for flat data."""

    def __init__(self, dom_dim, codom_dim, num_layers=3, hidden_size=32, spectral_norm=False):
        super().__init__()

        self.dom_dim = dom_dim
        self.codom_dim = codom_dim

        self._listify_hypers(num_layers, {"hidden_size": hidden_size})

        layers = []
        prev_size = dom_dim
        for size in self.hidden_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.SiLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, codom_dim))

        self.net = nn.Sequential(*layers)

        if spectral_norm:
            self._apply_spectral_norm()


class ImageSmoothMap(SmoothMap):
    """Smooth map for image data."""

    def __init__(self, dom_shape, codom_dim, num_layers=3, hidden_channels=32, kernel_size=3,
                 pool_size=2, spectral_norm=False):
        super().__init__()

        self.dom_shape = dom_shape
        self.dom_dim = int(np.prod(dom_shape))
        self.codom_dim = codom_dim

        self._listify_hypers(
            num_layers,
            {
                "hidden_channels": hidden_channels,
                "kernel_size": kernel_size,
                "pool_size": pool_size
            }
        )

        prev_channels, height, width = dom_shape
        assert height == width, "Network assumes square image"
        del width

        layers = []

        for channels, k, p in zip(self.hidden_channels, self.kernel_size, self.pool_size):
            layers.append(nn.Conv2d(prev_channels, hidden_channels, k))
            layers.append(nn.SiLU())
            height = self._get_new_height(height, k, 1) # Get height after conv

            if p is not None:
                layers.append(nn.AvgPool2d(p, ceil_mode=True))
                height = self._get_new_height(height, p, p, ceil=True) # Get height after pool

            prev_channels = channels

        layers.extend([
            nn.Flatten(),
            nn.Linear(channels*height**2, codom_dim),
        ])

        self.net = nn.Sequential(*layers)

        if spectral_norm:
            self._apply_spectral_norm()

    @staticmethod
    def _get_new_height(height, kernel, stride, ceil=False):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        if ceil:
            return math.ceil((height - kernel)/stride + 1)
        else:
            return math.floor((height - kernel)/stride + 1)


class ImageTransposeSmoothMap(SmoothMap):
    """Smooth map for image data."""

    def __init__(self, dom_dim, codom_shape, num_layers=3, hidden_channels=32, kernel_size=3,
                 spectral_norm=False):
        super().__init__()

        self.dom_dim = dom_dim
        self.codom_shape = codom_shape
        self.codom_dim = int(np.prod(codom_shape))

        self._listify_hypers(
            num_layers,
            {
                "hidden_channels": hidden_channels,
                "kernel_size": kernel_size,
            }
        )

        next_channels, height, width = codom_shape
        assert height == width, "Network assumes square image"
        del width

        layers = []

        # Loop from final to first layer to track dimensions
        for channels, k in zip(reversed(self.hidden_channels), reversed(self.kernel_size)):
            layers.insert(0, nn.ConvTranspose2d(hidden_channels, next_channels, k))
            layers.insert(1, nn.SiLU())

            height = self._get_new_height(height, k, 1) # Get height after conv

            next_channels = channels

        layers.insert(0, nn.Unflatten(1, (channels, height, height)))
        layers.insert(0, nn.Linear(dom_dim, channels*height**2))

        self.net = nn.Sequential(*layers)

        if spectral_norm:
            self._apply_spectral_norm()

    @staticmethod
    def _get_new_height(height, kernel, stride, ceil=False):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        if ceil:
            return math.ceil((height - kernel)/stride + 1)
        else:
            return math.floor((height - kernel)/stride + 1)



class UNetSmoothMap(UNet, SmoothMap):
    """Smooth map based on UNet."""

    def __init__(self, dom_shape, out_channels, size_factor=1, final_kernel_size=1,
                 spectral_norm=False, activation=nn.SiLU, fc_dim=None):
        SmoothMap.__init__(self)

        self.final_kernel_size = final_kernel_size
        self.dom_shape = dom_shape
        self.dom_dim = int(np.prod(dom_shape))
        unet_out_dim = out_channels * self._out_dim(dom_shape[1]) * self._out_dim(dom_shape[2])

        if fc_dim is not None:
            self.fc = nn.Linear(unet_out_dim, fc_dim)
            self.codom_dim = fc_dim
        else:
            self.codom_dim = unet_out_dim

        self._unet_setup(dom_shape[0], out_channels, size_factor, activation)
        self.flatten = nn.Flatten()

        if spectral_norm:
            self._apply_spectral_norm()

    def _unet_setup(self, in_channels, out_channels, size_factor, act):
        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList(
            [self.DoubleConvCustom(in_channels, int(64*size_factor), act=act)]
            + [self.DoubleConvCustom(int(i * size_factor), int(o * size_factor), act=act)
               for i, o in [(64, 128), (128, 256), (256, 512)]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = self.DoubleConvCustom(
            int(512 * size_factor), int(1024 * size_factor), act=act)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList(
            [UpSample(int(i * size_factor), int(o * size_factor)) for i, o in
             [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList(
            [self.DoubleConvCustom(int(i * size_factor), int(o * size_factor), act=act) for i, o in
             [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(int(64 * size_factor), out_channels,
                                    kernel_size=self.final_kernel_size)

    def forward(self, x):
        x = super().forward(x) # UNet pass
        x = self.flatten(x)
        if hasattr(self, "fc"):
            x = self.fc(x)
        return x

    def _out_dim(self, in_dim):
        # Compute output height or width based on input height or width
        dim = in_dim

        for _ in range(4):
            dim //= 2

        for _ in range(4):
            dim *= 2

        dim -= self.final_kernel_size - 1
        return dim

    class DoubleConvCustom(DoubleConvolution):
        def __init__(self, *args, act, **kwargs):
            super().__init__(*args, **kwargs)
            self.act1 = act()
            self.act2 = act()


class ResNetSmoothMap(SmoothMap):

    variants = {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    }

    def __init__(self, dom_shape, codom_dim, spectral_norm=False, variant="resnet18"):
        super().__init__()

        self.dom_shape = dom_shape
        self.dom_dim = int(np.prod(dom_shape))
        self.codom_dim = codom_dim

        assert variant in self.variants
        net_cls = getattr(torchvision.models, variant)
        # Can't use batchnorm with functorch
        self.net = net_cls(num_classes=codom_dim, norm_layer=nn.Identity)

        if spectral_norm:
            self._apply_spectral_norm()

    def forward(self, x):
        if self.dom_shape[0] == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.net(x)


class ConvNet2FC(SmoothMap):
    """Encoder used in NAE paper"""
    def __init__(self, dom_shape, codom_dim, nh=8, nh_mlp=512, spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super().__init__()

        in_channels, _, _ = self.dom_shape = dom_shape
        out_channels = self.codom_dim = codom_dim

        self.conv1 = nn.Conv2d(in_channels, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_channels, kernel_size=1, bias=True)

        if spectral_norm:
            self._apply_spectral_norm()

        layers = [self.conv1,
                  nn.ReLU(),
                  self.conv2,
                  nn.ReLU(),
                  self.max1,
                  self.conv3,
                  nn.ReLU(),
                  self.conv4,
                  nn.ReLU(),
                  self.max2,
                  self.conv5,
                  nn.ReLU(),
                  self.conv6,
                  nn.Flatten()]

        self.net = nn.Sequential(*layers)


class DeConvNet2(SmoothMap):
    """Decoder used in NAE paper"""
    def __init__(self, dom_dim, codom_shape, nh=8, spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super().__init__()

        in_channels = self.dom_dim = dom_dim
        out_channels, _, _ = self.codom_shape = codom_shape

        self.conv1 = nn.ConvTranspose2d(in_channels, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_channels, kernel_size=3, bias=True)

        if spectral_norm:
            self._apply_spectral_norm()

    def forward(self, x):
        x = x[..., None, None]
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        return x
