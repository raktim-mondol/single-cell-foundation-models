"""
GRN-Decoder VAE — Training Script
====================================
Train the GRN-constrained VAE and evaluate:
  - ELBO convergence
  - Edge recovery precision/recall vs ground-truth GRN
  - In silico TF knockout consistency
  - Calibrated interval coverage
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import GRNDecoderVAE
from utils import (
    generate_synthetic_grn_data,
    evaluate_calibration,
    set_seed,
)


class CountDataset(Dataset):
    def __init__(self, counts: np.ndarray):
        self.counts = torch.tensor(counts, dtype=torch.float32)

    def __len__(self):
        return self.counts.size(0)

    def __getitem__(self, idx):
        return self.counts[idx]


def train_epoch(model, loader, optimizer, device, kl_weight=1.0):
    model.train()
    totals = {"loss": 0, "recon_loss": 0, "kl_loss": 0, "l1_loss": 0}
    n = 0
    for counts in loader:
        counts = counts.to(device)
        out = model(counts)
        loss = out["recon_loss"] + kl_weight * out["kl_loss"] + \
               model.decoder.l1_lambda * out["l1_loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = counts.size(0)
        for k in totals:
            totals[k] += out[k].item() * bs
        n += bs
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    for counts in loader:
        counts = counts.to(device)
        out = model(counts)
        total_loss += out["loss"].item() * counts.size(0)
        n += counts.size(0)
    return total_loss / n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_genes",     type=int,   default=200)
    p.add_argument("--n_cells",     type=int,   default=2000)
    p.add_argument("--d_model",     type=int,   default=64)
    p.add_argument("--d_latent",    type=int,   default=16)
    p.add_argument("--n_enc_layers",type=int,   default=2)
    p.add_argument("--grn_density", type=float, default=0.03,
                   help="Fraction of non-zero edges in synthetic GRN")
    p.add_argument("--l1_lambda",   type=float, default=0.01)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--kl_warmup",   type=int,   default=10,
                   help="Epochs to warm up KL weight to 1")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--save_dir",    type=str,   default="checkpoints/grn_vae")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Synthetic GRN data ----
    data = generate_synthetic_grn_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        grn_density=args.grn_density,
        seed=args.seed,
    )
    grn_adj = data["grn_adj"]
    counts  = data["counts"]

    print(f"GRN edges: {(grn_adj != 0).sum()} / {args.n_genes**2}")

    dataset = CountDataset(counts)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = GRNDecoderVAE(
        n_genes=args.n_genes,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_enc_layers=args.n_enc_layers,
        grn_adj=grn_adj,
        l1_lambda=args.l1_lambda,
        use_zinb=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GRN-Decoder VAE parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(args.epochs):
        # KL warm-up
        kl_w = min(1.0, (epoch + 1) / args.kl_warmup)

        tr = train_epoch(model, train_loader, optimizer, device, kl_weight=kl_w)
        vl = val_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1:03d}  "
            f"loss={tr['loss']:.4f}  recon={tr['recon_loss']:.4f}  "
            f"kl={tr['kl_loss']:.4f}  l1={tr['l1_loss']:.4f}  "
            f"val_loss={vl:.4f}  kl_w={kl_w:.2f}"
        )

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_dir / "best.pt")

    print(f"\nBest val loss: {best_val:.4f}")

    # ---- Edge recovery ----
    print("\n--- Edge Recovery vs Ground-Truth GRN ---")
    er = model.edge_recovery_precision_recall(grn_adj, threshold=0.05)
    print(f"Precision: {er['precision']:.4f}  Recall: {er['recall']:.4f}  F1: {er['f1']:.4f}")

    # ---- In silico TF knockout ----
    print("\n--- In Silico TF Knockout (gene 0) ---")
    test_counts = torch.tensor(counts[:16], dtype=torch.float32, device=device)
    normal_pred = model.in_silico_knockout(test_counts, tf_gene_idx=-1)  # no KO
    ko_pred     = model.in_silico_knockout(test_counts, tf_gene_idx=0)

    delta_ko = (ko_pred - normal_pred).mean(0).cpu().numpy()
    n_down = (delta_ko < -0.1).sum()
    n_up   = (delta_ko >  0.1).sum()
    print(f"KO of gene 0 → {n_down} genes down-regulated, {n_up} up-regulated")

    # ---- Calibration ----
    print("\n--- Calibration Check ---")
    model.eval()
    calib = evaluate_calibration(model, counts[:200], device)
    print(f"Expected coverage at 95% CI: 0.950")
    print(f"Observed coverage:           {calib['coverage_95']:.4f}")
    print(f"ECE: {calib['ece']:.4f}")


if __name__ == "__main__":
    main()
