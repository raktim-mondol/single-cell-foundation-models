"""
SpaceTime-scFM — Utilities
"""

import random
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.manual_seed_all(seed)


def generate_synthetic_spatial_data(
    n_cells: int = 2000,
    n_genes: int = 500,
    spatial_dim: int = 2,
    n_niches: int = 5,
    spatial_frac: float = 0.6,
    pt_frac: float = 0.4,
    seed: int = 42,
) -> Dict:
    """
    Generate synthetic spatial transcriptomics data.

    Spatial structure: n_niches Gaussian clusters in 2D/3D space.
    Each niche has a distinct gene expression signature.
    """
    rng = np.random.default_rng(seed)

    # Niche centres
    centres = rng.uniform(-2, 2, size=(n_niches, spatial_dim))

    # Assign cells to niches
    niche_labels = rng.integers(0, n_niches, size=n_cells)

    # Spatial coordinates
    coords = np.stack([
        centres[niche_labels[i]] + rng.normal(0, 0.3, size=spatial_dim)
        for i in range(n_cells)
    ]).astype(np.float32)

    # Niche-specific gene signatures
    niche_means = rng.exponential(2.0, size=(n_niches, n_genes)).astype(np.float32)

    # Expression
    expr = np.stack([
        rng.negative_binomial(
            n=2, p=0.5, size=n_genes
        ).astype(np.float32) + niche_means[niche_labels[i]]
        for i in range(n_cells)
    ])
    expr = np.log1p(expr).astype(np.float32)

    # Pseudotime (linear from x-coordinate as a proxy)
    pseudotime = (coords[:, 0] - coords[:, 0].min()) / \
                 (coords[:, 0].max() - coords[:, 0].min() + 1e-8)
    pseudotime = pseudotime.astype(np.float32)

    # Mosaic: randomly drop modalities
    has_spatial = rng.random(size=n_cells) < spatial_frac
    has_pt      = rng.random(size=n_cells) < pt_frac

    return {
        "expr":        expr,
        "coords":      coords,
        "pseudotime":  pseudotime,
        "has_spatial": has_spatial,
        "has_pt":      has_pt,
        "niche_labels": niche_labels,
    }


def evaluate_niche_prediction(
    embeddings: np.ndarray,
    coords: Optional[np.ndarray],
    niche_labels: np.ndarray,
    cv: int = 5,
) -> float:
    """
    Evaluate how well cell embeddings predict tissue niche identity
    using logistic regression cross-validation.
    """
    clf = LogisticRegression(max_iter=500, C=1.0)
    scores = cross_val_score(clf, embeddings, niche_labels, cv=cv, scoring="accuracy")
    return float(scores.mean())
