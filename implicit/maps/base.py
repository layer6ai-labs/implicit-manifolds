from abc import abstractmethod

import torch.nn as nn


class Map(nn.Module):
    """Base class for all smooth maps."""

    @abstractmethod
    def forward(self, x):
        pass

    def inverse(self, x):
        """Some maps may have inverses, left-inverse, or right-inverses"""
        raise NotImplementedError("This map has no inverse of any kind.")

    def __matmul__(self, other):
        """Define composition of maps by (f @ g)(x) := f(g(x))"""
        return CompositeMap([other, self])


class CompositeMap(Map):
    """Composite of multiple smooth maps.

    Args:
        maps: an iterable of maps which will be composed in order; ie. [a, b, c] will be composed
            into x |-> c(b(a(x)))
    """

    def __init__(self, maps):
        super().__init__()
        self.maps = nn.ModuleList(maps)

    def forward(self, x):
        for mapping in self.maps:
            x = mapping.forward(x)
        return x

    def inverse(self, x):
        for mapping in reversed(self.maps):
            x = mapping.inverse(x)
        return x
