"""
CausalCellFM — Training Script
================================
Trains on perturb-seq data with a causal held-out evaluation scheme.

Data format expected:
  - ctrl_expr   : [N, n_genes]  log1p-normalised control expression
  - pert_expr   : [N, n_genes]  observed post-perturbation expression
  - pert_genes  : [N, K]        gene indices of perturbation targets
  - pert_dirs   : [N, K]        direction (0=KO, 1=WT, 2=OE)
  - pert_mags   : [N, K]        magnitude
  - de_mask     : [N, n_genes]  bool — differentially expressed genes
  - batch_ids   : [N]           batch / donor integer labels
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import CausalCellFM
from utils import (
    compute_perturbation_metrics,
    generate_synthetic_perturbseq,
    set_seed,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerturbSeqDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.ctrl_expr  = torch.tensor(data["ctrl_expr"],  dtype=torch.float32)
        self.pert_expr  = torch.tensor(data["pert_expr"],  dtype=torch.float32)
        self.pert_genes = torch.tensor(data["pert_genes"], dtype=torch.long)
        self.pert_dirs  = torch.tensor(data["pert_dirs"],  dtype=torch.long)
        self.pert_mags  = torch.tensor(data["pert_mags"],  dtype=torch.float32)
        self.de_mask    = torch.tensor(data["de_mask"],    dtype=torch.bool)
        self.batch_ids  = torch.tensor(data["batch_ids"],  dtype=torch.long)

    def __len__(self):
        return self.ctrl_expr.size(0)

    def __getitem__(self, idx):
        return {
            "ctrl_expr":  self.ctrl_expr[idx],
            "pert_expr":  self.pert_expr[idx],
            "pert_genes": self.pert_genes[idx],
            "pert_dirs":  self.pert_dirs[idx],
            "pert_mags":  self.pert_mags[idx],
            "de_mask":    self.de_mask[idx],
            "batch_ids":  self.batch_ids[idx],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: CausalCellFM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0, "mse_de": 0, "kl_loss": 0, "inv_loss": 0}
    n = 0

    for batch in loader:
        ctrl   = batch["ctrl_expr"].to(device)
        pert   = batch["pert_expr"].to(device)
        pg     = batch["pert_genes"].to(device)
        pd     = batch["pert_dirs"].to(device)
        pm     = batch["pert_mags"].to(device)
        de     = batch["de_mask"].to(device)
        bids   = batch["batch_ids"].to(device)

        out = model(ctrl, pg, pd, pm,
                    pert_expr=pert, de_mask=de, batch_labels=bids)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = ctrl.size(0)
        for k in totals:
            if k in out:
                totals[k] += out[k].item() * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: CausalCellFM,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_pred, all_true, all_ctrl, all_de = [], [], [], []

    for batch in loader:
        ctrl  = batch["ctrl_expr"].to(device)
        pert  = batch["pert_expr"].to(device)
        pg    = batch["pert_genes"].to(device)
        pd    = batch["pert_dirs"].to(device)
        pm    = batch["pert_mags"].to(device)
        de    = batch["de_mask"]

        out = model(ctrl, pg, pd, pm)
        all_pred.append(out["pred_expr"].cpu().numpy())
        all_true.append(pert.cpu().numpy())
        all_ctrl.append(ctrl.cpu().numpy())
        all_de.append(de.numpy())

    pred  = np.concatenate(all_pred, axis=0)
    true  = np.concatenate(all_true, axis=0)
    ctrl_ = np.concatenate(all_ctrl, axis=0)
    de    = np.concatenate(all_de, axis=0)

    return compute_perturbation_metrics(pred, true, ctrl_, de)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_genes",  type=int, default=1000)
    p.add_argument("--n_cells",  type=int, default=3000)
    p.add_argument("--d_model",  type=int, default=128)
    p.add_argument("--n_heads",  type=int, default=4)
    p.add_argument("--enc_layers", type=int, default=3)
    p.add_argument("--dec_layers", type=int, default=3)
    p.add_argument("--n_batches",  type=int, default=5)
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints/causalcell")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Synthetic perturb-seq data
    data = generate_synthetic_perturbseq(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    dataset = PerturbSeqDataset(data)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = CausalCellFM(
        n_genes=args.n_genes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        n_batches=args.n_batches,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CausalCellFM parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_pearson = -1.0
    for epoch in range(args.epochs):
        train_stats = train_epoch(model, train_loader, optimizer, device)
        val_stats   = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1:03d}  "
            f"loss={train_stats['loss']:.4f}  "
            f"mse_de={train_stats['mse_de']:.4f}  "
            f"Pearson_DE={val_stats['pearson_de']:.4f}  "
            f"Pearson_all={val_stats['pearson_all']:.4f}"
        )

        if val_stats["pearson_de"] > best_pearson:
            best_pearson = val_stats["pearson_de"]
            torch.save(model.state_dict(), save_dir / "best.pt")

    print(f"\nBest DE Pearson: {best_pearson:.4f}")

    # ---- Causal direction test ----
    print("\n--- Causal direction test (gene 0) ---")
    ctrl_sample = torch.tensor(
        data["ctrl_expr"][:16], dtype=torch.float32, device=device
    )
    result = model.causal_direction_test(ctrl_sample, gene_idx=0,
                                          n_genes=args.n_genes, device=device)
    print(f"CRISPRa vs CRISPRi correlation: {result['oe_ko_correlation']:.4f}")
    print(f"Expected sign: {result['expected_sign']}")


if __name__ == "__main__":
    main()
