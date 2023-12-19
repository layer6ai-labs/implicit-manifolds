"""
This package converts several `nflows`-style `Transform` classes into `Map` classes.

This allows us to write our own `Map` functionalities such as composition and Jacobian-related
operations while taking advantage of the many diffeomorphisms implemented in `nflows`.
"""
from .maps import FlatDiffeomorphism, MultiscaleEmbedding, LULinear, Project