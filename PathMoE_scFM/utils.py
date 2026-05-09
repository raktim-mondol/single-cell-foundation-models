"""
PathMoE-scFM — Utilities
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dummy_pathway_matrix(n_genes: int, n_experts: int) -> np.ndarray:
    """
    Create a random binary gene-pathway membership matrix.
    Each gene belongs to ~3 pathways on average.
    Replace this with actual Reactome / MSigDB membership in real use.
    """
    rng = np.random.default_rng(0)
    matrix = rng.binomial(1, p=3.0 / n_experts, size=(n_genes, n_experts)).astype(
        np.float32
    )
    # Ensure every gene has at least one pathway
    for i in range(n_genes):
        if matrix[i].sum() == 0:
            j = rng.integers(0, n_experts)
            matrix[i, j] = 1.0
    return matrix


def load_anndata(h5ad_path: str):
    """
    Load an AnnData .h5ad file.
    Returns (counts, labels, n_genes, n_classes).
    Requires scanpy and anndata installed.
    """
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        raise ImportError("Install anndata and scanpy: pip install anndata scanpy")

    adata = sc.read_h5ad(h5ad_path)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    counts = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)

    # Encode cell-type labels
    if "cell_type" in adata.obs.columns:
        labels_str = adata.obs["cell_type"].values
    else:
        labels_str = adata.obs.iloc[:, 0].values

    unique_labels = list(dict.fromkeys(labels_str))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels_str])
    n_classes = len(unique_labels)

    return counts.astype(np.float32), labels, counts.shape[1], n_classes


def compute_pathway_recovery(model, gene_pathway_matrix: np.ndarray) -> Dict:
    """
    Evaluate whether the MoE gating assigns high weight to correct
    pathway experts for each gene token.

    Computes precision@1 (top-1 routed expert matches a known pathway)
    and recall@5.
    """
    device = next(model.parameters()).device
    n_genes, n_experts = gene_pathway_matrix.shape

    # Create identity "cells" — one gene expressed per cell
    precision_at_1_list = []
    recall_at_5_list = []

    model.eval()
    with torch.no_grad():
        for gene_idx in range(min(n_genes, 200)):  # subsample for speed
            counts = torch.zeros(1, n_genes, device=device)
            counts[0, gene_idx] = 10.0  # single gene expressed

            tokens, _ = model.tokeniser(counts)
            # Forward through first block to get gate logits
            B, S, D = tokens.shape
            cls = model.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, tokens], dim=1)
            x_gene = x[:, 1 + gene_idx, :]  # [1, D]

            # Get gate logits for this gene token
            gate_logits = model.blocks[0].moe.gate(x_gene)  # [1, n_experts]
            gate_probs = torch.softmax(gate_logits, dim=-1).squeeze(0)  # [n_experts]

            top1 = gate_probs.argmax().item()
            top5 = gate_probs.topk(5).indices.tolist()

            known_pathways = np.where(gene_pathway_matrix[gene_idx] > 0)[0]
            if len(known_pathways) == 0:
                continue

            p1 = 1.0 if top1 in known_pathways else 0.0
            r5 = len(set(top5) & set(known_pathways)) / len(known_pathways)

            precision_at_1_list.append(p1)
            recall_at_5_list.append(r5)

    return {
        "precision_at_1": float(np.mean(precision_at_1_list)) if precision_at_1_list else 0.0,
        "recall_at_5": float(np.mean(recall_at_5_list)) if recall_at_5_list else 0.0,
    }
