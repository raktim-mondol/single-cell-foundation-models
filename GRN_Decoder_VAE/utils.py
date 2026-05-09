"""
GRN-Decoder VAE — Utilities
"""

import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_grn_data(
    n_cells: int = 2000,
    n_genes: int = 200,
    n_tfs: int = 20,
    grn_density: float = 0.03,
    seed: int = 42,
) -> Dict:
    """
    Generate synthetic expression data governed by a random GRN.

    TFs (first n_tfs genes) regulate downstream target genes.
    Expression of each target is a nonlinear function of its regulators.
    """
    rng = np.random.default_rng(seed)

    # Build sparse signed GRN adjacency
    grn_adj = np.zeros((n_genes, n_genes), dtype=np.float32)
    for g in range(n_tfs, n_genes):  # targets are non-TF genes
        n_regs = rng.poisson(grn_density * n_tfs)
        regs   = rng.choice(n_tfs, size=min(n_regs, n_tfs), replace=False)
        signs  = rng.choice([-1.0, 1.0], size=len(regs))
        for reg, sign in zip(regs, signs):
            grn_adj[g, reg] = sign

    # Simulate TF expression independently
    tf_expr = rng.gamma(2.0, scale=2.0, size=(n_cells, n_tfs)).astype(np.float32)

    # Simulate target expression as regulated function of TFs
    target_expr = np.zeros((n_cells, n_genes - n_tfs), dtype=np.float32)
    for g_idx, g in enumerate(range(n_tfs, n_genes)):
        regs = np.where(grn_adj[g, :n_tfs] != 0)[0]
        if len(regs) == 0:
            target_expr[:, g_idx] = rng.gamma(1.0, scale=1.0, size=n_cells).astype(np.float32)
        else:
            signs  = grn_adj[g, regs]
            signal = (tf_expr[:, regs] * signs).sum(axis=1)  # linear combo
            mu     = np.exp(signal * 0.3).clip(0.1, 20)
            target_expr[:, g_idx] = rng.negative_binomial(
                n=2, p=2 / (2 + mu)
            ).astype(np.float32)

    counts = np.concatenate([tf_expr, target_expr], axis=1)
    counts = np.clip(counts, 0, None)

    return {
        "counts":  counts.astype(np.float32),
        "grn_adj": grn_adj,
        "n_tfs":   n_tfs,
    }


@torch.no_grad()
def evaluate_calibration(
    model: nn.Module,
    counts: np.ndarray,
    device: torch.device,
    n_samples: int = 50,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Estimate calibration by drawing posterior samples and checking
    whether true counts fall within the predicted credible interval.

    We approximate coverage using Monte Carlo samples of z.
    """
    model.train()  # enable dropout for MC sampling
    counts_t = torch.tensor(counts, dtype=torch.float32, device=device)
    log_counts = torch.log1p(counts_t)
    n_cells, n_genes = counts.shape

    # Draw posterior samples
    samples = []
    for _ in range(n_samples):
        mu_z, log_var_z = model.encoder(log_counts)
        z = model.encoder.reparameterise(mu_z, log_var_z)
        dec = model.decoder(z)
        pred_counts = torch.exp(dec["log_mu"])  # [B, n_genes]
        samples.append(pred_counts.cpu().numpy())

    samples = np.stack(samples, axis=0)  # [n_samples, B, n_genes]

    alpha = 1.0 - confidence
    lower = np.quantile(samples, alpha / 2,  axis=0)  # [B, n_genes]
    upper = np.quantile(samples, 1 - alpha / 2, axis=0)

    true = counts
    in_interval = ((true >= lower) & (true <= upper)).astype(float)
    coverage = float(in_interval.mean())

    # Expected Calibration Error (binned by predicted mean)
    pred_mean = samples.mean(axis=0).flatten()
    true_flat  = true.flatten()
    n_bins = 10
    bin_edges = np.quantile(pred_mean, np.linspace(0, 1, n_bins + 1))
    ece = 0.0
    for i in range(n_bins):
        mask = (pred_mean >= bin_edges[i]) & (pred_mean <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc_in = in_interval.flatten()[mask].mean()
        ece += abs(acc_in - confidence) * mask.mean()

    model.eval()
    return {"coverage_95": coverage, "ece": float(ece)}
