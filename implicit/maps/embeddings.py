"""
Smooth, right-invertible maps R^n -> R^m (with m <= n).

Optimized correctly, they can embed a d-dimensional submanifold of R^n into R^m (with d <= m <= n).
"""
from torch.nn import functional as F

from .base import CompositeMap
from .nflows import FlatDiffeomorphism, MultiscaleEmbedding, LULinear, Project


class FlatEmbedding(CompositeMap):
    """Flow-like image embedding for flat data."""

    def __init__(self,
                 dom_dim,
                 codom_dim,
                 hidden_size,
                 num_layers,
                 num_blocks_per_layer,
                 include_linear=True,
                 coupling_type="affine",
                 **coupling_kwargs):
        self.dom_dim = dom_dim
        self.codom_dim = codom_dim

        assert coupling_type in ("affine", "additive", "rational-quadratic")
        diffeomorphism = FlatDiffeomorphism(
            dom_dim,
            hidden_size,
            num_layers,
            num_blocks_per_layer,
            include_linear,
            **coupling_kwargs,
        )
        proj = Project(dom_dim, codom_dim)

        super().__init__([diffeomorphism, proj])


class ImageEmbedding(CompositeMap):
    """Multiscale flow-like image embedding.

    This map is based on the multiscale composite transforms of `nflows`, except a certain number
    of channels is removed at each scale to make it an embedding. An LU-decomposed linear layer
    is applied to the output, and the result is projected down to the desired codomain dimension.
    """

    def __init__(self,
                 dom_shape,
                 codom_dim,
                 squeeze_factors,
                 projection_channels,
                 hidden_channels,
                 num_layers_per_level,
                 num_blocks_per_layer,
                 num_levels=None,
                 activation=F.silu,
                 dropout_probability=0.0,
                 batch_norm_within_layers=False,
                 linear_cache=False,
                 coupling_type="affine",
                 **coupling_kwargs):
        self.dom_shape = dom_shape
        self.codom_dim = codom_dim


        assert coupling_type in ("affine", "additive", "rational-quadratic")

        multiscale = MultiscaleEmbedding(
            dom_shape,
            squeeze_factors,
            projection_channels,
            hidden_channels,
            num_layers_per_level,
            num_blocks_per_layer,
            num_levels,
            activation,
            dropout_probability,
            batch_norm_within_layers,
            linear_cache,
            coupling_type,
            **coupling_kwargs,
        )
        linear = LULinear(multiscale.output_size)
        proj = Project(multiscale.output_size, codom_dim)

        super().__init__([multiscale, linear, proj])