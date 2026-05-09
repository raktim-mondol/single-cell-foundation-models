"""
Atlas-Streamer: Continual Learning for Atlas Updates

This package provides a continual learning framework for single-cell atlas
updates using self-distillation and importance-weighted replay.
"""

from .model import AtlasStreamer, SimpleScFMBackbone

__version__ = "0.1.0"
__all__ = ["AtlasStreamer", "SimpleScFMBackbone"]