"""
CausalCellFM — Utilities
"""

import random
from typing import Dict

import numpy as np
from scipy.stats import pearsonr


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.manual_seed_all(seed)


def generate_synthetic_perturbseq(
    n_cells: int = 3000,
    n_genes: int = 1000,
    n_batches: int = 5,
    max_pert_genes: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generates a synthetic perturb-seq dataset for testing.

    For each cell:
      - Draw a control expression from a negative binomial distribution.
      - Apply a random perturbation (KO a few genes) by zeroing their expression
        and up-regulating downstream targets stochastically.
      - Mark the differentially expressed genes as DE mask.
    """
    rng = np.random.default_rng(seed)

    # Control expression
    ctrl_expr = np.log1p(
        rng.negative_binomial(n=2, p=0.4, size=(n_cells, n_genes)).astype(np.float32)
    )

    # Perturbation specification
    K = max_pert_genes
    pert_genes = rng.integers(0, n_genes, size=(n_cells, K)).astype(np.int64)
    pert_dirs  = rng.integers(0, 3, size=(n_cells, K)).astype(np.int64)   # 0=KO,1=WT,2=OE
    pert_mags  = rng.uniform(0.5, 1.0, size=(n_cells, K)).astype(np.float32)

    # Simulate perturbed expression
    pert_expr = ctrl_expr.copy()
    de_mask   = np.zeros((n_cells, n_genes), dtype=bool)

    for i in range(n_cells):
        for j in range(K):
            gidx = pert_genes[i, j]
            direction = pert_dirs[i, j]
            mag = pert_mags[i, j]

            if direction == 0:    # KO — reduce target
                pert_expr[i, gidx] *= (1.0 - mag * 0.9)
            elif direction == 2:  # OE — amplify target
                pert_expr[i, gidx] *= (1.0 + mag * 2.0)
            # direction == 1 → no change

            de_mask[i, gidx] = True

            # Add some downstream DE noise (±30% of 5 random genes)
            downstream = rng.integers(0, n_genes, size=5)
            noise = rng.uniform(-0.3, 0.3, size=5)
            pert_expr[i, downstream] = np.clip(
                pert_expr[i, downstream] + noise, 0, None
            )
            de_mask[i, downstream] = True

    batch_ids = rng.integers(0, n_batches, size=n_cells).astype(np.int64)

    return {
        "ctrl_expr":  ctrl_expr.astype(np.float32),
        "pert_expr":  pert_expr.astype(np.float32),
        "pert_genes": pert_genes,
        "pert_dirs":  pert_dirs,
        "pert_mags":  pert_mags,
        "de_mask":    de_mask,
        "batch_ids":  batch_ids,
    }


def compute_perturbation_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray,
    de_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Standard perturb-seq evaluation metrics.

    Returns Pearson correlation on:
      - all genes
      - DE genes only (more informative, less zero-inflated)
    """
    delta_pred = pred - ctrl
    delta_true = true - ctrl

    # Flatten
    dp_flat = delta_pred.flatten()
    dt_flat = delta_true.flatten()
    pearson_all, _ = pearsonr(dp_flat, dt_flat)

    # DE-gene Pearson
    de_pred_vals = delta_pred[de_mask]
    de_true_vals = delta_true[de_mask]
    if len(de_pred_vals) > 1:
        pearson_de, _ = pearsonr(de_pred_vals, de_true_vals)
    else:
        pearson_de = float("nan")

    # MSE
    mse_all = float(np.mean((delta_pred - delta_true) ** 2))
    mse_de  = float(np.mean((de_pred_vals - de_true_vals) ** 2)) if len(de_pred_vals) > 1 else float("nan")

    return {
        "pearson_all": float(pearson_all),
        "pearson_de":  float(pearson_de),
        "mse_all":     mse_all,
        "mse_de":      mse_de,
    }
