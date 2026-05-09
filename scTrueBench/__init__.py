"""
scTrueBench: Causal Benchmark Suite

This package provides a comprehensive benchmark suite for evaluating
single-cell foundation models across causal reasoning, calibration,
and wet-lab validation axes.
"""

from .benchmark import (
    BenchmarkData,
    ScTrueBench,
    WetLabRegistry,
    WetLabPrediction,
    WetLabOutcome,
)

__version__ = "0.1.0"
__all__ = [
    "BenchmarkData",
    "ScTrueBench",
    "WetLabRegistry",
    "WetLabPrediction",
    "WetLabOutcome",
]