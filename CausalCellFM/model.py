"""
CausalCellFM: Counterfactual Perturbation Foundation Model
===========================================================
An encoder-decoder transformer trained with a causal loss against
held-out perturb-seq experiments.

Key ideas:
  - DE-gene-weighted MSE loss (not inflated by zeros)
  - Batch/donor invariance penalty
  - Encoder takes (control_cell, perturbation_tokens)
  - Decoder emits predicted perturbed transcriptome
  - Causal direction test: CRISPRa vs CRISPRi predictions must anti-correlate

Gaps addressed:
  A  – Correlation, not causation
  F  – Wet-lab grounded benchmarks
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Perturbation Tokeniser
# ---------------------------------------------------------------------------

class PerturbationTokeniser(nn.Module):
    """
    Encodes a perturbation specification as a dense vector.

    A perturbation is a list of (gene_idx, direction, magnitude) triples.
    direction ∈ {-1, 0, +1}  (knockout, wild-type, overexpression)
    magnitude ∈ [0, 1]        (relative strength)
    """

    def __init__(self, n_genes: int, d_model: int = 256, max_pert_genes: int = 10):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.max_pert_genes = max_pert_genes

        self.gene_embed = nn.Embedding(n_genes + 1, d_model, padding_idx=n_genes)
        self.dir_embed = nn.Embedding(3, d_model)   # 0=KO, 1=WT, 2=OE
        self.mag_proj = nn.Linear(1, d_model)
        self.combine = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        pert_genes: torch.Tensor,    # [B, max_pert_genes]  gene indices
        pert_dirs: torch.Tensor,     # [B, max_pert_genes]  direction label
        pert_mags: torch.Tensor,     # [B, max_pert_genes]  magnitude [0,1]
    ) -> torch.Tensor:
        """Returns [B, max_pert_genes, d_model]."""
        g = self.gene_embed(pert_genes)              # [B, K, d]
        d = self.dir_embed(pert_dirs)                # [B, K, d]
        m = self.mag_proj(pert_mags.unsqueeze(-1))   # [B, K, d]
        x = self.combine(torch.cat([g, d, m], dim=-1))
        return self.norm(x)


# ---------------------------------------------------------------------------
# 2. Cell Encoder (control expression → latent)
# ---------------------------------------------------------------------------

class CellEncoder(nn.Module):
    """
    Light-weight transformer encoder that maps a control cell's
    expression + perturbation tokens to a joint latent sequence.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model

        # Simple linear projection for expression (no binning — we want magnitude)
        self.expr_proj = nn.Linear(n_genes, d_model)
        self.expr_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        expr: torch.Tensor,          # [B, n_genes]  log1p-normalised expression
        pert_tokens: torch.Tensor,   # [B, K, d_model] from PerturbationTokeniser
    ) -> torch.Tensor:
        """Returns [B, 1 + K, d_model] latent sequence."""
        B = expr.size(0)

        # Compress expression to single cell token
        cell_tok = self.expr_norm(self.expr_proj(expr))  # [B, d_model]
        cell_tok = cell_tok.unsqueeze(1)                 # [B, 1, d_model]

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, cell_tok, pert_tokens], dim=1)  # [B, 2+K, d]
        x = self.transformer(x)
        return x


# ---------------------------------------------------------------------------
# 3. Expression Decoder
# ---------------------------------------------------------------------------

class ExpressionDecoder(nn.Module):
    """
    Cross-attention decoder that generates predicted post-perturbation
    expression from the encoder latent.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes

        # Gene query embeddings — one per output gene
        self.gene_queries = nn.Embedding(n_genes, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output: predict log1p expression for each gene
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory : [B, seq, d_model]  encoder output
        Returns predicted log1p expression [B, n_genes]
        """
        B = memory.size(0)
        gene_ids = torch.arange(self.n_genes, device=memory.device)
        queries = self.gene_queries(gene_ids)         # [n_genes, d_model]
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_genes, d_model]

        out = self.transformer(queries, memory)       # [B, n_genes, d_model]
        pred = self.out_proj(out).squeeze(-1)         # [B, n_genes]
        return pred


# ---------------------------------------------------------------------------
# 4. Batch Invariance Discriminator
# ---------------------------------------------------------------------------

class BatchDiscriminator(nn.Module):
    """
    Predicts batch/donor label from the cell latent.
    Adversarial training: encoder is trained to fool this discriminator,
    making the latent batch-invariant.
    """

    def __init__(self, d_model: int, n_batches: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_batches),
        )

    def forward(self, cell_latent: torch.Tensor) -> torch.Tensor:
        return self.net(cell_latent)


# ---------------------------------------------------------------------------
# 5. Full CausalCellFM
# ---------------------------------------------------------------------------

class CausalCellFM(nn.Module):
    """
    Encoder-decoder causal perturbation model.

    Loss components:
      α · MSE_DE   — MSE only on differentially expressed (DE) genes
      β · KL_delta — KL between predicted and observed expression delta
      γ · inv_loss — batch invariance (via adversarial gradient reversal)
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        enc_layers: int = 4,
        dec_layers: int = 4,
        n_batches: int = 10,
        max_pert_genes: int = 5,
        alpha: float = 2.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.pert_tokeniser = PerturbationTokeniser(n_genes, d_model, max_pert_genes)
        self.encoder = CellEncoder(n_genes, d_model, n_heads, enc_layers, dropout)
        self.decoder = ExpressionDecoder(n_genes, d_model, n_heads, dec_layers, dropout)
        self.batch_disc = BatchDiscriminator(d_model, n_batches, dropout)

    def forward(
        self,
        ctrl_expr: torch.Tensor,       # [B, n_genes]  log1p control expression
        pert_genes: torch.Tensor,       # [B, K]        gene indices
        pert_dirs: torch.Tensor,        # [B, K]        direction labels
        pert_mags: torch.Tensor,        # [B, K]        magnitudes
        pert_expr: Optional[torch.Tensor] = None,  # [B, n_genes] observed perturbed
        de_mask: Optional[torch.Tensor] = None,    # [B, n_genes] bool, DE genes
        batch_labels: Optional[torch.Tensor] = None,  # [B]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with 'pred_expr', 'loss', and sub-loss components.
        """
        # Encode
        pert_tokens = self.pert_tokeniser(pert_genes, pert_dirs, pert_mags)
        latent = self.encoder(ctrl_expr, pert_tokens)   # [B, 2+K, d]
        cell_latent = latent[:, 0, :]                   # CLS token [B, d]

        # Decode
        pred_expr = self.decoder(latent)                # [B, n_genes]

        out = {"pred_expr": pred_expr}

        if pert_expr is None:
            return out

        # ---- DE-weighted MSE ----
        delta_pred = pred_expr - ctrl_expr              # predicted delta
        delta_true = pert_expr - ctrl_expr              # observed delta

        if de_mask is not None:
            # Only on DE genes
            mse_de = F.mse_loss(
                delta_pred[de_mask], delta_true[de_mask]
            )
        else:
            mse_de = F.mse_loss(delta_pred, delta_true)

        # ---- KL between predicted and observed delta distributions ----
        # Treat as Gaussian with diagonal covariance
        eps = 1e-6
        pred_mu = delta_pred.mean(dim=-1)
        pred_sigma = delta_pred.std(dim=-1).clamp(min=eps)
        true_mu = delta_true.mean(dim=-1)
        true_sigma = delta_true.std(dim=-1).clamp(min=eps)

        kl_loss = (
            torch.log(pred_sigma / true_sigma)
            + (true_sigma**2 + (true_mu - pred_mu)**2) / (2 * pred_sigma**2)
            - 0.5
        ).mean()

        # ---- Batch invariance loss ----
        inv_loss = torch.tensor(0.0, device=ctrl_expr.device)
        if batch_labels is not None:
            batch_pred = self.batch_disc(cell_latent)
            # Adversarial: we want encoder to maximise batch confusion
            # (gradient reversal trick approximated by negating loss for encoder)
            inv_loss = F.cross_entropy(batch_pred, batch_labels)

        total_loss = (
            self.alpha * mse_de
            + self.beta * kl_loss.clamp(min=0)
            - self.gamma * inv_loss   # negative: encoder maximises batch confusion
        )

        out.update({
            "loss": total_loss,
            "mse_de": mse_de,
            "kl_loss": kl_loss,
            "inv_loss": inv_loss,
            "cell_latent": cell_latent,
        })
        return out

    def causal_direction_test(
        self,
        ctrl_expr: torch.Tensor,
        gene_idx: int,
        n_genes: int,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Causal direction test:
        Predict CRISPRa (OE) and CRISPRi (KO) for a single gene.
        The two predicted deltas should anti-correlate.
        """
        B = ctrl_expr.size(0)
        K = 1

        def make_pert(direction: int):
            pg = torch.full((B, K), gene_idx, dtype=torch.long, device=device)
            pd = torch.full((B, K), direction, dtype=torch.long, device=device)
            pm = torch.ones(B, K, device=device) * 0.9
            return pg, pd, pm

        self.eval()
        with torch.no_grad():
            pg_oe, pd_oe, pm_oe = make_pert(2)   # overexpression
            pg_ko, pd_ko, pm_ko = make_pert(0)   # knockout

            out_oe = self(ctrl_expr, pg_oe, pd_oe, pm_oe)
            out_ko = self(ctrl_expr, pg_ko, pd_ko, pm_ko)

            delta_oe = (out_oe["pred_expr"] - ctrl_expr).mean(0).cpu().numpy()
            delta_ko = (out_ko["pred_expr"] - ctrl_expr).mean(0).cpu().numpy()

        # Anti-correlation indicates causal consistency
        corr = float(np.corrcoef(delta_oe, delta_ko)[0, 1])
        return {"oe_ko_correlation": corr, "expected_sign": "negative"}
