"""
`Transforms` in the original `nflows` structure.
"""
import numpy as np
import torch
from torch.nn import functional as F
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform, MultiscaleCompositeTransform, Transform
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    AdditiveCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform
)
from nflows.transforms.conv import OneByOneConvolution
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.reshape import SqueezeTransform


_coupling_dict = {
    "affine": AffineCouplingTransform,
    "additive": AdditiveCouplingTransform,
    "rational-quadratic": PiecewiseRationalQuadraticCouplingTransform,
}


class FlatDiffeomorphismTransform(CompositeTransform):
    """Simple flow transform designed to act on flat data"""

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        include_linear=True,
        num_bins=8,
        tail_bound=1.0,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        coupling_constructor=PiecewiseRationalQuadraticCouplingTransform,
    ):
        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            coupling_transform = coupling_constructor(
                mask=mask,
                transform_net_create_fn=create_resnet,
                tails="linear",
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            layers.append(coupling_transform)
            mask *= -1

            if include_linear:
                linear_transform = CompositeTransform([
                    RandomPermutation(features=features),
                    LULinear(features, identity_init=True)])
                layers.append(linear_transform)

        super().__init__(layers)


class MultiscaleEmbeddingTransform(MultiscaleCompositeTransform):

    def __init__(self,
                 input_shape,
                 squeeze_factors,
                 projection_channels,
                 hidden_channels,
                 num_layers_per_level,
                 num_blocks_per_layer,
                 num_levels=None,
                 activation=F.relu,
                 dropout_probability=0.0,
                 batch_norm_within_layers=False,
                 linear_cache=False,
                 coupling_type="affine",
                 **coupling_kwargs):

        super().__init__(num_levels)

        def create_resnet(in_features, out_features):
            return nets.ConvResidualNet(
                in_features,
                out_features,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        self.output_size = 0
        num_levels = num_levels if num_levels is not None else len(projection_channels)
        assert num_levels > 0, "You need at least one level!"
        assert len(squeeze_factors) == len(projection_channels)
        try:
            coupling_constructor = _coupling_dict[coupling_type]
        except KeyError:
            raise ValueError(f"`coupling_type` must be in {_coupling_dict.keys()}")


        for level in range(num_levels):
            channels, height, width = input_shape
            squeeze_factor = squeeze_factors[level]
            project_channels = projection_channels[level]
            layers = []

            squeeze_transform = SqueezeTransform(factor=squeeze_factor)
            layers.append(squeeze_transform)

            channels *= squeeze_factor**2
            height //= squeeze_factor
            width //= squeeze_factor

            mask = torch.ones(channels)
            mask[::2] = -1

            for _ in range(num_layers_per_level):
                coupling_transform = coupling_constructor(
                    mask=mask,
                    transform_net_create_fn=create_resnet,
                    **coupling_kwargs
                )
                layers.append(coupling_transform)
                mask *= -1

                linear_transform = OneByOneConvolution(channels, using_cache=linear_cache)
                layers.append(linear_transform)

            project_transform = ProjectTransform(channels, project_channels)
            layers.append(project_transform)
            channels = project_channels

            input_shape = self.add_transform(
                CompositeTransform(layers),
                (channels, height, width)
            )

            if input_shape is not None:
                self.output_size += int(channels * height * width - np.prod(input_shape))
            else:
                self.output_size += int(channels * height * width)


class ProjectTransform(Transform):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return inputs[:,:self.out_channels,...], logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)

        if inputs.dim() == 4:
            outputs = F.pad(inputs, pad=(0, 0, 0, 0, 0, self.in_channels - self.out_channels))
        else:
            outputs = F.pad(inputs, pad=(0, self.in_channels - self.out_channels))
        return outputs, logabsdet