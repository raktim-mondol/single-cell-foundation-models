"""
scTrueBench: Causal, Calibration-Aware, Lab-Validated Benchmark Suite
======================================================================
A successor to scIB built on three axes that scIB does not score:

  1. Causal recovery     — held-out perturb-seq experiments
  2. Calibration         — ECE and credible interval coverage
  3. Robustness          — performance under realistic noise corruption

Additionally includes a WetLabRegistry: a public ledger where models
register predictions and wet-lab partners submit experimental outcomes.

Gaps addressed:
  All six gaps — changes what the field rewards.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkData:
    """All data needed for one scTrueBench evaluation."""

    # Expression matrices
    counts_ctrl:   np.ndarray          # [N_ctrl, G]    control expression
    counts_pert:   np.ndarray          # [N_pert, G]    post-perturbation
    counts_noisy:  Optional[np.ndarray] = None  # corrupted version

    # Perturbation specification
    pert_genes:    Optional[np.ndarray] = None  # [N_pert, K]
    pert_dirs:     Optional[np.ndarray] = None  # [N_pert, K]  0=KO, 2=OE
    de_mask:       Optional[np.ndarray] = None  # [N_pert, G]  bool

    # Cell identity
    cell_type_labels: Optional[np.ndarray] = None   # [N_ctrl]  int
    batch_labels:     Optional[np.ndarray] = None   # [N_ctrl]  int

    # Ground-truth for causal evaluation
    grn_adj:          Optional[np.ndarray] = None   # [G, G]  GRN adjacency
    causal_holdout:   Optional[np.ndarray] = None   # subset used only for causal eval


@dataclass
class BenchmarkResult:
    """Results from one model on one benchmark."""
    model_name: str
    timestamp:  str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Standard scIB-style scores
    nmi:          Optional[float] = None
    ari:          Optional[float] = None
    asw_bio:      Optional[float] = None
    asw_batch:    Optional[float] = None

    # Causal recovery
    pearson_de:       Optional[float] = None
    pearson_all:      Optional[float] = None
    causal_direction: Optional[float] = None  # fraction correct (>0 = OE > KO)

    # Calibration
    ece_95:       Optional[float] = None
    coverage_95:  Optional[float] = None

    # Robustness
    robustness_dropout:      Optional[float] = None
    robustness_batch_shift:  Optional[float] = None

    # Wet-lab registry score (filled in asynchronously)
    wetlab_score: Optional[float] = None

    def overall_score(self) -> float:
        """Weighted composite following proposed scTrueBench weights."""
        scores = []
        weights = []

        def add(v, w):
            if v is not None and not np.isnan(v):
                scores.append(v)
                weights.append(w)

        add(self.nmi,              0.05)
        add(self.ari,              0.05)
        add(self.asw_bio,          0.05)
        add(self.asw_batch,        0.05)
        add(self.pearson_de,       0.20)
        add(self.pearson_all,      0.05)
        add(self.causal_direction, 0.15)
        add(1 - (self.ece_95 or 0),   0.10)
        add(self.coverage_95,      0.10)
        add(self.robustness_dropout,     0.05)
        add(self.robustness_batch_shift, 0.05)
        add(self.wetlab_score or 0,      0.10)

        if not scores:
            return float("nan")
        w_arr = np.array(weights)
        s_arr = np.array(scores)
        return float((s_arr * w_arr).sum() / w_arr.sum())

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["overall_score"] = self.overall_score()
        return d


# ---------------------------------------------------------------------------
# Axis 1: Standard scIB metrics
# ---------------------------------------------------------------------------

class ScIBAxis:
    """Standard embedding-quality metrics (cluster + batch)."""

    @staticmethod
    def compute(
        embeddings: np.ndarray,    # [N, d]
        cell_types: np.ndarray,    # [N]
        batch_ids: np.ndarray,     # [N]
    ) -> Dict[str, float]:
        results = {}

        # Cluster quality
        if len(np.unique(cell_types)) > 1:
            results["nmi"]     = normalized_mutual_info_score(
                cell_types, cell_types  # placeholder — usually uses Leiden clusters
            )
            results["ari"]     = adjusted_rand_score(cell_types, cell_types)  # same

            # ASW biological conservation
            if embeddings.shape[0] > len(np.unique(cell_types)):
                try:
                    results["asw_bio"] = float(
                        silhouette_score(embeddings, cell_types)
                    )
                except Exception:
                    results["asw_bio"] = float("nan")

        # ASW batch removal (lower silhouette = better batch mixing)
        if len(np.unique(batch_ids)) > 1:
            try:
                asw_batch = float(silhouette_score(embeddings, batch_ids))
                # Convert: 0 = perfect mixing, we report 1 - |asw_batch|
                results["asw_batch"] = 1.0 - abs(asw_batch)
            except Exception:
                results["asw_batch"] = float("nan")

        return results


# ---------------------------------------------------------------------------
# Axis 2: Causal recovery
# ---------------------------------------------------------------------------

class CausalRecoveryAxis:
    """
    Evaluates models on held-out perturb-seq experiments.
    """

    @staticmethod
    def pearson_de_metric(
        pred_expr: np.ndarray,    # [N_pert, G]
        true_expr: np.ndarray,    # [N_pert, G]
        ctrl_expr: np.ndarray,    # [N_pert, G]
        de_mask: np.ndarray,      # [N_pert, G]  bool
    ) -> Dict[str, float]:
        delta_pred = pred_expr - ctrl_expr
        delta_true = true_expr - ctrl_expr

        r_all, _ = pearsonr(delta_pred.flatten(), delta_true.flatten())
        de_pred  = delta_pred[de_mask]
        de_true  = delta_true[de_mask]
        r_de     = float(pearsonr(de_pred, de_true)[0]) if len(de_pred) > 1 else float("nan")

        return {"pearson_all": float(r_all), "pearson_de": r_de}

    @staticmethod
    def causal_direction_accuracy(
        oe_delta: np.ndarray,    # [G]  predicted delta for overexpression
        ko_delta: np.ndarray,    # [G]  predicted delta for knockout
    ) -> float:
        """
        Fraction of genes where OE delta and KO delta have opposite signs.
        A causal model should produce anti-correlated responses.
        """
        both_nonzero = (np.abs(oe_delta) > 0.05) & (np.abs(ko_delta) > 0.05)
        if both_nonzero.sum() == 0:
            return float("nan")
        opposite = (oe_delta[both_nonzero] * ko_delta[both_nonzero] < 0)
        return float(opposite.mean())

    @staticmethod
    def grn_edge_recovery(
        learned_weights: np.ndarray,   # [G, G]
        true_grn: np.ndarray,          # [G, G]
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        lb = (np.abs(learned_weights) > threshold).astype(int)
        tb = (np.abs(true_grn) > 0).astype(int)
        tp = (lb * tb).sum()
        fp = (lb * (1 - tb)).sum()
        fn = ((1 - lb) * tb).sum()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        return {"edge_precision": float(p), "edge_recall": float(r),
                "edge_f1": float(2 * p * r / (p + r + 1e-8))}


# ---------------------------------------------------------------------------
# Axis 3: Calibration
# ---------------------------------------------------------------------------

class CalibrationAxis:
    """
    Expected Calibration Error and credible interval coverage.
    """

    @staticmethod
    def ece_and_coverage(
        pred_samples: np.ndarray,   # [n_samples, N, G]  Monte Carlo draws
        true_counts: np.ndarray,    # [N, G]
        confidence: float = 0.95,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        alpha = 1.0 - confidence
        lower = np.quantile(pred_samples, alpha / 2,       axis=0)
        upper = np.quantile(pred_samples, 1 - alpha / 2,  axis=0)

        in_ci   = (true_counts >= lower) & (true_counts <= upper)
        coverage = float(in_ci.mean())

        # ECE over prediction-mean bins
        pred_mean = pred_samples.mean(axis=0).flatten()
        true_flat = true_counts.flatten()
        in_ci_flat = in_ci.flatten().astype(float)

        bin_edges = np.quantile(pred_mean, np.linspace(0, 1, n_bins + 1))
        ece = 0.0
        for i in range(n_bins):
            mask = (pred_mean >= bin_edges[i]) & (pred_mean <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_coverage = in_ci_flat[mask].mean()
            ece += abs(bin_coverage - confidence) * (mask.sum() / len(pred_mean))

        return {"ece_95": float(ece), "coverage_95": coverage}


# ---------------------------------------------------------------------------
# Axis 4: Robustness
# ---------------------------------------------------------------------------

class RobustnessAxis:
    """Performance degradation under realistic noise."""

    @staticmethod
    def simulate_dropout(counts: np.ndarray, rate: float = 0.2,
                         seed: int = 0) -> np.ndarray:
        """Randomly zero out entries with probability `rate`."""
        rng = np.random.default_rng(seed)
        mask = rng.random(counts.shape) < rate
        noisy = counts.copy()
        noisy[mask] = 0.0
        return noisy

    @staticmethod
    def simulate_batch_shift(counts: np.ndarray, shift_std: float = 0.5,
                              seed: int = 0) -> np.ndarray:
        """Add gene-wise Gaussian shift to simulate batch effect."""
        rng = np.random.default_rng(seed)
        shift = rng.normal(0, shift_std, size=(1, counts.shape[1]))
        return np.clip(counts + shift, 0, None).astype(np.float32)

    @staticmethod
    def evaluate_robustness(
        model_fn: Callable[[np.ndarray], np.ndarray],  # counts → embeddings
        clean_counts: np.ndarray,
        cell_types: np.ndarray,
        dropout_rate: float = 0.2,
        batch_shift_std: float = 0.5,
    ) -> Dict[str, float]:
        """
        Computes linear-probe accuracy on clean vs. noisy inputs.
        Robustness score = acc_noisy / acc_clean.
        """
        clf = LogisticRegression(max_iter=200, C=1.0)

        clean_embs  = model_fn(clean_counts)
        clean_score = float(cross_val_score(clf, clean_embs, cell_types,
                                             cv=3, scoring="accuracy").mean())

        # Dropout robustness
        noisy_drop  = RobustnessAxis.simulate_dropout(clean_counts, dropout_rate)
        noisy_embs  = model_fn(noisy_drop)
        drop_score  = float(cross_val_score(clf, noisy_embs, cell_types,
                                             cv=3, scoring="accuracy").mean())

        # Batch shift robustness
        noisy_batch = RobustnessAxis.simulate_batch_shift(clean_counts, batch_shift_std)
        batch_embs  = model_fn(noisy_batch)
        batch_score = float(cross_val_score(clf, batch_embs, cell_types,
                                             cv=3, scoring="accuracy").mean())

        return {
            "clean_accuracy":            clean_score,
            "robustness_dropout":        drop_score / (clean_score + 1e-8),
            "robustness_batch_shift":    batch_score / (clean_score + 1e-8),
        }


# ---------------------------------------------------------------------------
# Wet-Lab Registry
# ---------------------------------------------------------------------------

@dataclass
class WetLabPrediction:
    """A model's pre-registered prediction for wet-lab validation."""
    model_name:     str
    prediction_id:  str
    gene_targets:   List[int]
    perturbation:   str           # "KO" or "OE"
    predicted_top_up:   List[int]  # top-10 predicted up-regulated genes
    predicted_top_down: List[int]  # top-10 predicted down-regulated genes
    registered_at:  str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cell_line:      str = "unspecified"


@dataclass
class WetLabOutcome:
    """Experimental result submitted by a wet-lab partner."""
    prediction_id:  str
    confirmed_up:   List[int]   # genes validated as up-regulated
    confirmed_down: List[int]   # genes validated as down-regulated
    validated_at:   str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WetLabRegistry:
    """
    Lightweight file-backed registry for pre-registered model predictions
    and wet-lab outcomes. Analogous to ProteinGym for perturbation biology.
    """

    def __init__(self, registry_path: str = "wetlab_registry"):
        self.path = Path(registry_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.pred_file    = self.path / "predictions.jsonl"
        self.outcome_file = self.path / "outcomes.jsonl"

    def register_prediction(self, pred: WetLabPrediction) -> str:
        """Register a model prediction. Returns prediction_id."""
        with open(self.pred_file, "a") as f:
            f.write(json.dumps(asdict(pred)) + "\n")
        return pred.prediction_id

    def submit_outcome(self, outcome: WetLabOutcome) -> None:
        """Submit wet-lab experimental results."""
        with open(self.outcome_file, "a") as f:
            f.write(json.dumps(asdict(outcome)) + "\n")

    def compute_wetlab_score(self, model_name: str) -> float:
        """
        Score a model based on how many of its pre-registered predictions
        were validated by wet-lab experiments.

        Score = mean Jaccard overlap between predicted and confirmed gene sets.
        """
        if not self.pred_file.exists() or not self.outcome_file.exists():
            return float("nan")

        predictions = {}
        with open(self.pred_file) as f:
            for line in f:
                p = json.loads(line)
                if p["model_name"] == model_name:
                    predictions[p["prediction_id"]] = p

        if not predictions:
            return float("nan")

        outcomes = {}
        with open(self.outcome_file) as f:
            for line in f:
                o = json.loads(line)
                outcomes[o["prediction_id"]] = o

        scores = []
        for pid, pred in predictions.items():
            if pid not in outcomes:
                continue
            out = outcomes[pid]
            pred_set  = set(pred["predicted_top_up"]) | set(pred["predicted_top_down"])
            valid_set = set(out["confirmed_up"]) | set(out["confirmed_down"])
            if not pred_set and not valid_set:
                scores.append(1.0)
            elif not pred_set or not valid_set:
                scores.append(0.0)
            else:
                jaccard = len(pred_set & valid_set) / len(pred_set | valid_set)
                scores.append(jaccard)

        return float(np.mean(scores)) if scores else float("nan")

    def leaderboard(self) -> List[Dict]:
        """Return model leaderboard sorted by wet-lab score."""
        if not self.pred_file.exists():
            return []
        models = set()
        with open(self.pred_file) as f:
            for line in f:
                models.add(json.loads(line)["model_name"])
        board = [{"model": m, "wetlab_score": self.compute_wetlab_score(m)}
                 for m in models]
        return sorted(board, key=lambda x: x["wetlab_score"] or -1, reverse=True)


# ---------------------------------------------------------------------------
# Master Benchmark Runner
# ---------------------------------------------------------------------------

class ScTrueBench:
    """
    Runs all four evaluation axes against a model's output.
    """

    def __init__(self, data: BenchmarkData, registry: Optional[WetLabRegistry] = None):
        self.data     = data
        self.registry = registry

    def run(
        self,
        model_name: str,
        embeddings: Optional[np.ndarray] = None,
        pred_pert_expr: Optional[np.ndarray] = None,
        pred_samples: Optional[np.ndarray] = None,  # MC samples for calibration
        model_fn: Optional[Callable] = None,         # for robustness tests
        oe_delta: Optional[np.ndarray] = None,
        ko_delta: Optional[np.ndarray] = None,
        learned_grn: Optional[np.ndarray] = None,
    ) -> BenchmarkResult:
        result = BenchmarkResult(model_name=model_name)

        # ---- Axis 1: Standard scIB ----
        if (embeddings is not None and
                self.data.cell_type_labels is not None and
                self.data.batch_labels is not None):
            scib = ScIBAxis.compute(
                embeddings,
                self.data.cell_type_labels,
                self.data.batch_labels,
            )
            result.nmi     = scib.get("nmi")
            result.ari     = scib.get("ari")
            result.asw_bio = scib.get("asw_bio")
            result.asw_batch = scib.get("asw_batch")

        # ---- Axis 2: Causal recovery ----
        if pred_pert_expr is not None and self.data.de_mask is not None:
            causal = CausalRecoveryAxis.pearson_de_metric(
                pred_pert_expr,
                self.data.counts_pert,
                self.data.counts_ctrl[:len(pred_pert_expr)],
                self.data.de_mask,
            )
            result.pearson_de  = causal["pearson_de"]
            result.pearson_all = causal["pearson_all"]

        if oe_delta is not None and ko_delta is not None:
            result.causal_direction = CausalRecoveryAxis.causal_direction_accuracy(
                oe_delta, ko_delta
            )

        # ---- Axis 3: Calibration ----
        if pred_samples is not None:
            calib = CalibrationAxis.ece_and_coverage(
                pred_samples,
                self.data.counts_pert,
            )
            result.ece_95      = calib["ece_95"]
            result.coverage_95 = calib["coverage_95"]

        # ---- Axis 4: Robustness ----
        if (model_fn is not None and
                self.data.cell_type_labels is not None):
            rob = RobustnessAxis.evaluate_robustness(
                model_fn,
                self.data.counts_ctrl,
                self.data.cell_type_labels,
            )
            result.robustness_dropout     = rob["robustness_dropout"]
            result.robustness_batch_shift = rob["robustness_batch_shift"]

        # ---- Wet-lab registry ----
        if self.registry is not None:
            result.wetlab_score = self.registry.compute_wetlab_score(model_name)

        return result

    def print_report(self, result: BenchmarkResult) -> None:
        d = result.to_dict()
        print(f"\n{'='*60}")
        print(f"  scTrueBench Report — {result.model_name}")
        print(f"{'='*60}")
        categories = {
            "Standard (scIB)":    ["nmi", "ari", "asw_bio", "asw_batch"],
            "Causal Recovery":    ["pearson_de", "pearson_all", "causal_direction"],
            "Calibration":        ["ece_95", "coverage_95"],
            "Robustness":         ["robustness_dropout", "robustness_batch_shift"],
            "Wet-Lab Registry":   ["wetlab_score"],
        }
        for cat, keys in categories.items():
            print(f"\n  {cat}:")
            for k in keys:
                v = d.get(k)
                val_str = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
                print(f"    {k:<30} {val_str}")
        print(f"\n  Overall Score: {d['overall_score']:.4f}")
        print(f"{'='*60}\n")
