"""
CausalCellFM: Counterfactual Perturbation Foundation Model

This package provides a causal inference framework for single-cell perturbation
data that learns causal relationships rather than mere correlations.
"""

from .model import CausalCellFM

__version__ = "0.1.0"
__all__ = ["CausalCellFM"]