"""
GRN-Decoder VAE: GRN-Constrained Generative Foundation Model

This package provides a VAE with GRN-constrained decoder for biologically
valid generative modeling of single-cell data.
"""

from .model import GRNDecoderVAE

__version__ = "0.1.0"
__all__ = ["GRNDecoderVAE"]