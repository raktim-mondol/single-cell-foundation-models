"""
scTrueBench — Demo Runner
===========================
Demonstrates all four benchmark axes against two toy models:
  - "Random"    : random embeddings (baseline floor)
  - "Linear PCA": simple PCA + linear regression (competitive scIB baseline)

Shows how a real scFM would be plugged in.
"""

import argparse
import uuid

import numpy as np
import torch
from sklearn.decomposition import PCA

from benchmark import (
    BenchmarkData,
    ScTrueBench,
    WetLabRegistry,
    WetLabPrediction,
    WetLabOutcome,
)
from utils import generate_benchmark_dataset, set_seed


def random_model_fn(counts: np.ndarray) -> np.ndarray:
    """Baseline: random embeddings."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((counts.shape[0], 32)).astype(np.float32)


def pca_model_fn(counts: np.ndarray, n_components: int = 32) -> np.ndarray:
    """Baseline: PCA embeddings."""
    pca = PCA(n_components=min(n_components, counts.shape[0], counts.shape[1]))
    return pca.fit_transform(np.log1p(counts)).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_cells",  type=int, default=500)
    p.add_argument("--n_genes",  type=int, default=200)
    p.add_argument("--n_types",  type=int, default=5)
    p.add_argument("--n_batches",type=int, default=3)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--registry_path", type=str, default="wetlab_registry_demo")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Generating synthetic benchmark dataset …")
    data = generate_benchmark_dataset(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_cell_types=args.n_types,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    bench_data = BenchmarkData(
        counts_ctrl=data["counts_ctrl"],
        counts_pert=data["counts_pert"],
        pert_genes=data["pert_genes"],
        pert_dirs=data["pert_dirs"],
        de_mask=data["de_mask"],
        cell_type_labels=data["cell_types"],
        batch_labels=data["batch_ids"],
        grn_adj=data.get("grn_adj"),
    )

    # ---- Demo wet-lab registry ----
    registry = WetLabRegistry(args.registry_path)

    # Register predictions for PCA model
    pid = str(uuid.uuid4())[:8]
    pred = WetLabPrediction(
        model_name="PCA",
        prediction_id=pid,
        gene_targets=[0, 1],
        perturbation="KO",
        predicted_top_up=[10, 11, 12, 15, 20, 25, 30, 35, 40, 50],
        predicted_top_down=[5, 6, 7, 8, 9, 13, 14, 16, 17, 18],
        cell_line="K562",
    )
    registry.register_prediction(pred)

    # Simulate a wet-lab outcome (partial overlap)
    outcome = WetLabOutcome(
        prediction_id=pid,
        confirmed_up=[10, 11, 13, 15, 22],
        confirmed_down=[5, 6, 8, 9, 19],
    )
    registry.submit_outcome(outcome)

    bench = ScTrueBench(bench_data, registry=registry)

    # ---- Evaluate models ----
    for model_name, model_fn in [("Random", random_model_fn), ("PCA", pca_model_fn)]:
        print(f"\nEvaluating: {model_name}")

        embs = model_fn(bench_data.counts_ctrl)

        # Trivial perturbation prediction: copy control (worst case)
        pred_pert = bench_data.counts_ctrl[:len(bench_data.counts_pert)].copy()

        # Mock MC samples (for calibration demo — just gaussian noise around pred)
        rng = np.random.default_rng(0)
        pred_samples = np.stack([
            pred_pert + rng.normal(0, 0.5, pred_pert.shape)
            for _ in range(30)
        ])

        # Mock OE/KO deltas
        oe_delta = rng.normal(0.1,  0.3, bench_data.counts_ctrl.shape[1])
        ko_delta = rng.normal(-0.1, 0.3, bench_data.counts_ctrl.shape[1])

        result = bench.run(
            model_name=model_name,
            embeddings=embs,
            pred_pert_expr=pred_pert,
            pred_samples=pred_samples,
            model_fn=model_fn,
            oe_delta=oe_delta,
            ko_delta=ko_delta,
        )

        bench.print_report(result)

    # ---- Leaderboard ----
    print("Wet-Lab Leaderboard:")
    for entry in registry.leaderboard():
        print(f"  {entry['model']:<20} wet_lab_score={entry['wetlab_score']:.4f}")

    print("\nHow to plug in a real scFM:")
    print("  1. Define model_fn(counts: np.ndarray) -> np.ndarray (returns embeddings)")
    print("  2. Provide pred_pert_expr from your model's perturbation prediction head")
    print("  3. Provide pred_samples (Monte Carlo posterior draws) for calibration")
    print("  4. Register predictions in WetLabRegistry before running wet-lab experiments")


if __name__ == "__main__":
    main()
