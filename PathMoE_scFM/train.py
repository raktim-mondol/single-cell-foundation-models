"""
PathMoE-scFM — Training Script
================================
Pretrain on masked-gene-expression prediction, then fine-tune for
cell-type annotation with optional few-shot support.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import CellTypeClassifier, PathMoEscFM
from utils import (
    build_dummy_pathway_matrix,
    load_anndata,
    compute_pathway_recovery,
    set_seed,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SingleCellDataset(Dataset):
    """Wraps an AnnData count matrix for pretraining."""

    def __init__(self, counts: np.ndarray):
        self.counts = torch.tensor(counts, dtype=torch.float32)

    def __len__(self):
        return self.counts.size(0)

    def __getitem__(self, idx):
        return self.counts[idx]


class AnnotatedSingleCellDataset(Dataset):
    """For supervised fine-tuning (cell-type annotation)."""

    def __init__(self, counts: np.ndarray, labels: np.ndarray):
        self.counts = torch.tensor(counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.counts.size(0)

    def __getitem__(self, idx):
        return self.counts[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------------

def pretrain(
    model: PathMoEscFM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    lr: float,
    device: torch.device,
    save_dir: Path,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_loader)
    )

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            counts = batch.to(device)
            out = model(counts)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                counts = batch.to(device)
                out = model(counts)
                val_loss += out["loss"].item()
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1:03d}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_pretrain.pt")

    print(f"Best validation loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    backbone: PathMoEscFM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    n_epochs: int,
    lr: float,
    device: torch.device,
    save_dir: Path,
):
    model = CellTypeClassifier(backbone, n_classes, d_model=backbone.tokeniser.d_model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        total, correct = 0, 0
        for counts, labels in train_loader:
            counts, labels = counts.to(device), labels.to(device)
            logits = model(counts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for counts, labels in val_loader:
                counts, labels = counts.to(device), labels.to(device)
                logits = model(counts)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(
            f"FT Epoch {epoch+1:02d}/{n_epochs}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_finetune.pt")

    print(f"Best val accuracy: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation: pathway membership recovery
# ---------------------------------------------------------------------------

def evaluate_pathway_recovery(model: PathMoEscFM, gene_pathway_matrix: np.ndarray):
    """
    Checks that the gating distribution for each gene aligns with
    its known pathway memberships (precision@1 metric).
    """
    scores = compute_pathway_recovery(model, gene_pathway_matrix)
    print(f"Pathway recovery precision@1: {scores['precision_at_1']:.4f}")
    print(f"Pathway recovery recall@5:    {scores['recall_at_5']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PathMoE-scFM training")
    p.add_argument("--h5ad", type=str, default=None,
                   help="Path to AnnData .h5ad file (None = use synthetic demo)")
    p.add_argument("--n_genes", type=int, default=2000)
    p.add_argument("--n_cells", type=int, default=5000,
                   help="Number of synthetic cells (ignored if --h5ad given)")
    p.add_argument("--n_experts", type=int, default=50,
                   help="Number of pathway experts")
    p.add_argument("--top_k", type=int, default=4,
                   help="Active experts per token")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--pretrain_epochs", type=int, default=20)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints/pathmoe")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    if args.h5ad:
        counts, labels, n_genes, n_classes = load_anndata(args.h5ad)
    else:
        print("No h5ad provided — generating synthetic count data …")
        rng = np.random.default_rng(args.seed)
        n_genes = args.n_genes
        counts = rng.negative_binomial(
            n=2, p=0.5, size=(args.n_cells, n_genes)
        ).astype(np.float32)
        labels = rng.integers(0, 10, size=args.n_cells)
        n_classes = 10

    # ---- Pathway matrix (demo: random binary) ----
    gene_pathway_matrix = build_dummy_pathway_matrix(n_genes, args.n_experts)

    # ---- Datasets & loaders ----
    dataset = SingleCellDataset(counts)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- Model ----
    model = PathMoEscFM(
        n_genes=n_genes,
        n_experts=args.n_experts,
        top_k=args.top_k,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        gene_pathway_matrix=gene_pathway_matrix,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PathMoE-scFM parameters: {n_params:,}")

    # ---- Pretrain ----
    save_dir = Path(args.save_dir)
    pretrain(model, train_loader, val_loader,
             args.pretrain_epochs, args.lr, device, save_dir)

    # ---- Evaluate pathway recovery ----
    evaluate_pathway_recovery(model, gene_pathway_matrix)

    # ---- Fine-tune (cell-type annotation) ----
    ann_ds_train = AnnotatedSingleCellDataset(
        counts[:n_train], labels[:n_train]
    )
    ann_ds_val = AnnotatedSingleCellDataset(
        counts[n_train:], labels[n_train:]
    )
    ft_train = DataLoader(ann_ds_train, batch_size=args.batch_size, shuffle=True)
    ft_val = DataLoader(ann_ds_val, batch_size=args.batch_size)

    finetune(model, ft_train, ft_val, n_classes,
             args.finetune_epochs, args.lr, device, save_dir)


if __name__ == "__main__":
    main()
