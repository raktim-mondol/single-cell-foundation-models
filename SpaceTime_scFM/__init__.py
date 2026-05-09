"""
SpaceTime-scFM: Spatio-Temporal Multi-Modal Foundation Model

This package provides a multi-modal foundation model that treats spatial
coordinates and pseudotime as first-class tokens for comprehensive
cellular context understanding.
"""

from .model import SpaceTimescFM

__version__ = "0.1.0"
__all__ = ["SpaceTimescFM"]