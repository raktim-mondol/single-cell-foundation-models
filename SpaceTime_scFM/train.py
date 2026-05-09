"""
SpaceTime-scFM — Training Script
==================================
Pretrain with mosaic objectives:
  - All cells contribute to masked gene prediction
  - Cells with spatial coordinates contribute to masked-position prediction
  - Cells with pseudotime contribute to pseudotime regression
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from model import SpaceTimescFM
from utils import (
    generate_synthetic_spatial_data,
    evaluate_niche_prediction,
    set_seed,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpatioTemporalDataset(Dataset):
    def __init__(self, data):
        self.expr       = torch.tensor(data["expr"],       dtype=torch.float32)
        self.coords     = torch.tensor(data["coords"],     dtype=torch.float32) \
                          if data["coords"] is not None else None
        self.pseudotime = torch.tensor(data["pseudotime"], dtype=torch.float32) \
                          if data["pseudotime"] is not None else None
        self.has_spatial = data["has_spatial"]   # bool array [N]
        self.has_pt      = data["has_pt"]        # bool array [N]

    def __len__(self):
        return self.expr.size(0)

    def __getitem__(self, idx):
        item = {"expr": self.expr[idx]}
        if self.coords is not None and self.has_spatial[idx]:
            item["coords"] = self.coords[idx]
        if self.pseudotime is not None and self.has_pt[idx]:
            item["pseudotime"] = self.pseudotime[idx]
        return item


def collate_fn(batch):
    """Handle variable-presence modalities."""
    expr = torch.stack([b["expr"] for b in batch])
    out = {"expr": expr}

    # Spatial: only for items that have it
    coords_list = [b.get("coords") for b in batch]
    if any(c is not None for c in coords_list):
        # Pad missing with zeros; mask handled by has_spatial flag
        coords = torch.stack([
            c if c is not None else torch.zeros_like(coords_list[
                next(i for i, x in enumerate(coords_list) if x is not None)])
            for c in coords_list
        ])
        has_spatial = torch.tensor([c is not None for c in coords_list])
        out["coords"] = coords
        out["has_spatial"] = has_spatial

    pt_list = [b.get("pseudotime") for b in batch]
    if any(p is not None for p in pt_list):
        pt = torch.tensor([
            p.item() if p is not None else 0.5
            for p in pt_list
        ])
        has_pt = torch.tensor([p is not None for p in pt_list])
        out["pseudotime"] = pt
        out["has_pt"] = has_pt

    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, mask_spatial_prob=0.15):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        expr       = batch["expr"].to(device)
        coords     = batch.get("coords")
        pseudotime = batch.get("pseudotime")
        has_spatial = batch.get("has_spatial")

        if coords is not None:
            coords = coords.to(device)
            # Only pass coords for cells that actually have spatial data
            if has_spatial is not None:
                coords[~has_spatial] = 0.0

        if pseudotime is not None:
            pseudotime = pseudotime.to(device)

        # Randomly decide whether to mask spatial position
        do_mask_spatial = (
            coords is not None and np.random.random() < mask_spatial_prob
        )

        out = model(
            expr,
            coords=coords,
            pseudotime=pseudotime,
            mask_genes=True,
            mask_spatial=do_mask_spatial,
        )
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * expr.size(0)
        n += expr.size(0)

    return total_loss / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        expr       = batch["expr"].to(device)
        coords     = batch.get("coords")
        pseudotime = batch.get("pseudotime")
        if coords is not None:
            coords = coords.to(device)
        if pseudotime is not None:
            pseudotime = pseudotime.to(device)

        out = model(expr, coords=coords, pseudotime=pseudotime,
                    mask_genes=True, mask_spatial=False)
        total_loss += out["loss"].item() * expr.size(0)
        n += expr.size(0)
    return total_loss / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_genes",  type=int, default=500)
    p.add_argument("--n_cells",  type=int, default=2000)
    p.add_argument("--d_model",  type=int, default=128)
    p.add_argument("--n_heads",  type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--spatial_dim", type=int, default=2,
                   help="2 for Visium/Stereo-seq, 3 for 3D MERFISH")
    p.add_argument("--spatial_frac", type=float, default=0.6,
                   help="Fraction of cells with spatial coordinates")
    p.add_argument("--pt_frac",      type=float, default=0.4,
                   help="Fraction of cells with pseudotime")
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints/spacetime")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = generate_synthetic_spatial_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        spatial_dim=args.spatial_dim,
        spatial_frac=args.spatial_frac,
        pt_frac=args.pt_frac,
        seed=args.seed,
    )

    dataset = SpatioTemporalDataset(data)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              collate_fn=collate_fn)

    model = SpaceTimescFM(
        n_genes=args.n_genes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        spatial_dim=args.spatial_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SpaceTime-scFM parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        vl_loss = val_epoch(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}  train={tr_loss:.4f}  val={vl_loss:.4f}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

    print(f"\nBest val loss: {best_val:.4f}")

    # ---- Niche prediction evaluation ----
    print("\n--- Niche prediction evaluation ---")
    embs = model.get_cell_embeddings(
        torch.tensor(data["expr"], dtype=torch.float32, device=device),
        coords=torch.tensor(data["coords"], dtype=torch.float32, device=device)
        if data["coords"] is not None else None,
    )
    score = evaluate_niche_prediction(
        embs.cpu().numpy(),
        data["coords"] if data["coords"] is not None else None,
        data["niche_labels"],
    )
    print(f"Niche prediction accuracy: {score:.4f}")


if __name__ == "__main__":
    main()
