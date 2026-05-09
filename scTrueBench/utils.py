"""
scTrueBench — Utilities
"""

import random
from typing import Dict

import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.manual_seed_all(seed)


def generate_benchmark_dataset(
    n_cells: int = 500,
    n_genes: int = 200,
    n_cell_types: int = 5,
    n_batches: int = 3,
    n_pert_genes: int = 3,
    seed: int = 42,
) -> Dict:
    """
    Generate a synthetic dataset for scTrueBench evaluation.

    Returns control and perturbed expression matrices, DE masks,
    cell-type and batch labels, and a ground-truth sparse GRN.
    """
    rng = np.random.default_rng(seed)

    # Cell-type-specific expression means
    type_means = rng.exponential(2.0, size=(n_cell_types, n_genes)).astype(np.float32)
    cell_types = rng.integers(0, n_cell_types, size=n_cells)

    # Control expression
    counts_ctrl = np.stack([
        rng.negative_binomial(2, 0.5, n_genes).astype(np.float32)
        + type_means[cell_types[i]]
        for i in range(n_cells)
    ]).astype(np.float32)

    # Batch effects
    batch_ids = rng.integers(0, n_batches, size=n_cells)
    batch_shifts = rng.normal(0, 0.3, size=(n_batches, n_genes)).astype(np.float32)
    for i in range(n_cells):
        counts_ctrl[i] += batch_shifts[batch_ids[i]]
    counts_ctrl = np.clip(counts_ctrl, 0, None)

    # Perturbed expression (knock out random genes in each cell)
    pert_genes = rng.integers(0, n_genes, size=(n_cells, n_pert_genes)).astype(np.int64)
    pert_dirs  = rng.integers(0, 3, size=(n_cells, n_pert_genes)).astype(np.int64)

    counts_pert = counts_ctrl.copy()
    de_mask     = np.zeros((n_cells, n_genes), dtype=bool)

    for i in range(n_cells):
        for j in range(n_pert_genes):
            g   = pert_genes[i, j]
            dir = pert_dirs[i, j]
            if dir == 0:
                counts_pert[i, g] *= 0.1
            elif dir == 2:
                counts_pert[i, g] *= 3.0
            de_mask[i, g] = True
            # Downstream effects
            ds = rng.integers(0, n_genes, size=3)
            counts_pert[i, ds] += rng.normal(0, 0.5, 3)
            de_mask[i, ds] = True
    counts_pert = np.clip(counts_pert, 0, None).astype(np.float32)

    # Sparse GRN
    grn_adj = (rng.random((n_genes, n_genes)) < 0.02).astype(np.float32)
    np.fill_diagonal(grn_adj, 0)

    return {
        "counts_ctrl":  counts_ctrl,
        "counts_pert":  counts_pert,
        "cell_types":   cell_types.astype(np.int64),
        "batch_ids":    batch_ids.astype(np.int64),
        "pert_genes":   pert_genes,
        "pert_dirs":    pert_dirs,
        "de_mask":      de_mask,
        "grn_adj":      grn_adj,
    }
