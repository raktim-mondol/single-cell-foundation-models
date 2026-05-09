"""
PathMoE-scFM: Pathway-Aware Sparse Mixture-of-Experts Transformer

This package provides a pathway-aware foundation model for single-cell genomics
that uses biological pathway priors to guide model learning.
"""

from .model import (
    GeneTokeniser,
    PathwayExpert,
    PathwayMoE,
    PathMoETransformerBlock,
    PathMoEscFM,
    CellTypeClassifier,
)

__version__ = "0.1.0"
__all__ = [
    "GeneTokeniser",
    "PathwayExpert",
    "PathwayMoE",
    "PathMoETransformerBlock",
    "PathMoEscFM",
    "CellTypeClassifier",
]