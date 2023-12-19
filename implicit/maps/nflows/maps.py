"""
(Smooth) maps defined using underlying `nflows`-style `Transform` classes.
"""
from nflows import transforms

from ..base import Map
from .transforms import FlatDiffeomorphismTransform, MultiscaleEmbeddingTransform, ProjectTransform


def map_from_transform(transform_cls):

    class NFlowsMap(transform_cls, Map):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x):
            out, _ = super().forward(x)
            return out

        def inverse(self, x):
            out, _ = super().inverse(x)
            return out

    return NFlowsMap


LULinear = map_from_transform(transforms.LULinear)
MultiscaleEmbedding = map_from_transform(MultiscaleEmbeddingTransform)
Project = map_from_transform(ProjectTransform)
FlatDiffeomorphism = map_from_transform(FlatDiffeomorphismTransform)