"""
Atlas-Streamer — Utilities
"""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_streaming_releases(
    n_releases: int = 5,
    cells_per_release: int = 500,
    n_genes_init: int = 500,
    gene_growth: int = 50,
    n_cell_types_init: int = 8,
    new_types_per_release: int = 2,
    n_donors_per_release: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Simulate a series of CELLxGENE data releases.

    Each release:
      - Adds gene_growth new gene measurements
      - Introduces new_types_per_release previously unseen cell types
      - Has a fresh batch of donor IDs
    """
    rng = np.random.default_rng(seed)
    releases = []
    total_types = n_cell_types_init

    for r in range(n_releases):
        n_genes = n_genes_init + r * gene_growth
        n_types = total_types + r * new_types_per_release

        # Cell types for this release (include some old + some new)
        if r == 0:
            available_types = list(range(n_cell_types_init))
        else:
            new_type_ids = list(range(total_types, total_types + new_types_per_release))
            old_type_ids = list(range(total_types))
            available_types = old_type_ids + new_type_ids
            total_types += new_types_per_release

        cell_type_ids = rng.choice(available_types, size=cells_per_release)

        # Gene-expression means per cell type
        type_means = rng.exponential(2.0, size=(n_types, n_genes)).astype(np.float32)

        counts = np.stack([
            rng.negative_binomial(2, 0.5, n_genes).astype(np.float32)
            + type_means[cell_type_ids[i]]
            for i in range(cells_per_release)
        ])
        counts = np.log1p(counts).astype(np.float32)

        donor_ids = rng.integers(0, n_donors_per_release, size=cells_per_release)

        releases.append({
            "counts":     counts,
            "cell_types": cell_type_ids.astype(np.int64),
            "donor_ids":  donor_ids.astype(np.int64),
            "n_genes":    n_genes,
            "n_types":    n_types,
            "new_types":  list(range(total_types - new_types_per_release, total_types))
                          if r > 0 else [],
        })

    return releases


@torch.no_grad()
def evaluate_cell_type_accuracy(
    backbone: nn.Module,
    classifier: nn.Module,
    counts: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Evaluate linear-probe accuracy on held-out data."""
    backbone.eval()
    classifier.eval()
    all_correct, total = 0, 0
    for i in range(0, len(counts), batch_size):
        c = counts[i:i+batch_size].to(device)
        l = labels[i:i+batch_size].to(device)
        # Align gene dim
        if c.shape[1] < backbone.n_genes:
            pad = torch.zeros(c.shape[0], backbone.n_genes - c.shape[1], device=device)
            c = torch.cat([c, pad], dim=1)
        embs = backbone(c)
        preds = classifier(embs).argmax(dim=-1)
        all_correct += (preds == l).sum().item()
        total += l.size(0)
    return all_correct / total if total > 0 else 0.0
