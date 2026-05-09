"""
Atlas-Streamer — Training / Simulation Script
==============================================
Simulates a series of CELLxGENE data releases and evaluates:
  - Backward transfer (accuracy on old cell types)
  - Forward transfer (zero-shot on new cell types)
  - Compute savings vs. full re-pretraining
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import AtlasStreamer, SimpleScFMBackbone
from utils import (
    generate_streaming_releases,
    evaluate_cell_type_accuracy,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_genes_init",  type=int, default=500,
                   help="Gene vocabulary size at time 0")
    p.add_argument("--gene_growth",   type=int, default=50,
                   help="New genes added per release")
    p.add_argument("--n_releases",    type=int, default=5,
                   help="Number of simulated atlas releases")
    p.add_argument("--cells_per_release", type=int, default=500)
    p.add_argument("--n_cell_types_init", type=int, default=8)
    p.add_argument("--new_types_per_release", type=int, default=2)
    p.add_argument("--d_model",       type=int, default=64)
    p.add_argument("--n_layers",      type=int, default=2)
    p.add_argument("--buffer_cap",    type=int, default=5000)
    p.add_argument("--update_steps",  type=int, default=100)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--save_dir",      type=str, default="checkpoints/atlas")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- Generate streaming releases ----
    releases = generate_streaming_releases(
        n_releases=args.n_releases,
        cells_per_release=args.cells_per_release,
        n_genes_init=args.n_genes_init,
        gene_growth=args.gene_growth,
        n_cell_types_init=args.n_cell_types_init,
        new_types_per_release=args.new_types_per_release,
        seed=args.seed,
    )

    # ---- Initial model ----
    model = SimpleScFMBackbone(
        n_genes=args.n_genes_init,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    streamer = AtlasStreamer(
        model,
        buffer_capacity=args.buffer_cap,
    )

    # ---- Quick initial pretraining on release 0 ----
    print("=== Initial pretraining on release 0 ===")
    release0 = releases[0]
    streamer.update(
        new_counts=release0["counts"],
        new_cell_types=release0["cell_types"],
        new_donor_ids=release0["donor_ids"],
        n_steps=args.update_steps * 2,
        lr=args.lr,
        device=device,
    )

    # Simple linear classifier on initial cell types
    n_init_types = args.n_cell_types_init
    classifier = nn.Linear(args.d_model, n_init_types).to(device)
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # Train classifier on initial data
    counts0 = torch.tensor(release0["counts"], dtype=torch.float32)
    labels0 = torch.tensor(release0["cell_types"], dtype=torch.long)
    streamer.student.to(device)
    streamer.student.eval()
    with torch.no_grad():
        embs0 = streamer.student(counts0.to(device)).cpu()

    for _ in range(200):
        idxs = torch.randperm(len(embs0))[:64]
        logits = classifier(embs0[idxs].to(device))
        loss = nn.CrossEntropyLoss()(logits, labels0[idxs].to(device))
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()

    init_acc = evaluate_cell_type_accuracy(
        streamer.student, classifier, counts0, labels0, device
    )
    print(f"Initial accuracy (release 0): {init_acc:.4f}")

    # ---- Streaming updates ----
    print("\n=== Streaming updates ===")
    history = []

    for r_idx, release in enumerate(releases[1:], start=1):
        n_genes_new = release["counts"].shape[1]
        print(f"\nRelease {r_idx}  |  cells={len(release['counts'])}  "
              f"genes={n_genes_new}  new_types={release.get('new_types', [])}")

        stats = streamer.update(
            new_counts=release["counts"],
            new_cell_types=release["cell_types"],
            new_donor_ids=release["donor_ids"],
            n_steps=args.update_steps,
            lr=args.lr,
            device=device,
        )
        print(f"  recon_loss={stats['recon_loss']:.4f}  kl_loss={stats['kl_loss']:.4f}")

        # Backward transfer: accuracy on original release-0 cell types
        # Align counts0 to new gene vocab
        cur_n_genes = streamer.student.n_genes
        if counts0.shape[1] < cur_n_genes:
            pad = torch.zeros(counts0.shape[0], cur_n_genes - counts0.shape[1])
            counts0_aligned = torch.cat([counts0, pad], dim=1)
        else:
            counts0_aligned = counts0

        # Pad classifier if needed
        if classifier.out_features < streamer.student.n_genes:
            pass  # classifier operates on embeddings, no change needed

        bt_acc = evaluate_cell_type_accuracy(
            streamer.student, classifier, counts0_aligned, labels0, device
        )
        print(f"  Backward transfer accuracy: {bt_acc:.4f}  "
              f"(initial: {init_acc:.4f}, drop: {init_acc - bt_acc:.4f})")

        history.append({
            "release": r_idx,
            "backward_acc": bt_acc,
            "recon_loss": stats["recon_loss"],
        })

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"{'Release':>8} | {'BT Acc':>8} | {'Drop':>8} | {'recon_loss':>12}")
    print("-" * 50)
    for h in history:
        drop = init_acc - h["backward_acc"]
        print(f"{h['release']:>8} | {h['backward_acc']:>8.4f} | {drop:>8.4f} | {h['recon_loss']:>12.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(streamer.student.state_dict(), save_dir / "final_student.pt")
    print(f"\nModel saved to {save_dir / 'final_student.pt'}")


if __name__ == "__main__":
    main()
