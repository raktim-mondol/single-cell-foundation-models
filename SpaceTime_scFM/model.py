"""
SpaceTime-scFM: Spatio-Temporal Multi-Modal Foundation Model
=============================================================
Extends a standard scFM with two new token types:
  1. Position tokens  — 2D/3D spatial coordinates (Visium, MERFISH, Slide-tags)
     discretised into a learned positional codebook
  2. Pseudotime tokens — RNA-velocity or diffusion pseudotime, discretised

The attention operates over
  (gene, expression, position, pseudotime, modality, batch) tuples.

Pretraining objectives (mosaic-friendly — objectives only activate
when the relevant modality is present):
  a) Masked gene expression prediction
  b) Masked-position prediction: predict spatial neighbourhood bin
  c) Pseudotime regression on lineage-traced subsets

Gaps addressed:
  C  – No spatial or temporal context
  B  – Tissue-level biological priors
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Spatial Position Encoder
# ---------------------------------------------------------------------------

class SpatialPositionEncoder(nn.Module):
    """
    Maps continuous (x, y[, z]) coordinates to a d_model-dimensional vector.
    Trained with a contrastive loss so that spatially near cells have
    similar embeddings.

    Supports 2D (Visium/Stereo-seq) and 3D (3D MERFISH).
    """

    def __init__(self, d_model: int = 256, spatial_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.spatial_dim = spatial_dim

        # Fourier feature embedding for coordinates
        n_freqs = d_model // (2 * spatial_dim)
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0, n_freqs - 1, n_freqs),  # [n_freqs]
        )

        fourier_out = 2 * spatial_dim * n_freqs  # sin + cos per dim per freq
        self.proj = nn.Sequential(
            nn.Linear(fourier_out, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [B, spatial_dim]  normalised to [-1, 1]
        Returns [B, 2 * spatial_dim * n_freqs]
        """
        # [B, spatial_dim, 1] * [n_freqs] → [B, spatial_dim, n_freqs]
        phase = coords.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0) * math.pi
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return features.flatten(start_dim=1)  # [B, 2*spatial_dim*n_freqs]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [B, spatial_dim]  → [B, d_model]"""
        ff = self.fourier_features(coords)
        return self.norm(self.proj(ff))

    def contrastive_loss(
        self,
        coords: torch.Tensor,          # [B, spatial_dim]
        pos_emb: torch.Tensor,         # [B, d_model]
        temperature: float = 0.07,
        near_threshold: float = 0.1,   # fraction of tissue diameter
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss:
        cells within near_threshold of each other are positive pairs.
        """
        B = coords.size(0)
        # Pairwise distances
        dists = torch.cdist(coords.float(), coords.float())  # [B, B]
        pos_mask = (dists < near_threshold).float()
        pos_mask.fill_diagonal_(0)

        # Cosine similarity of embeddings
        emb_norm = F.normalize(pos_emb, dim=-1)
        sim = emb_norm @ emb_norm.T / temperature  # [B, B]

        # For each anchor, treat all positives as targets
        # Use sum of log-softmax over positives
        log_softmax = F.log_softmax(sim, dim=-1)  # [B, B]
        n_pos = pos_mask.sum(dim=-1).clamp(min=1)
        loss = -(pos_mask * log_softmax).sum(dim=-1) / n_pos
        return loss.mean()


# ---------------------------------------------------------------------------
# 2. Pseudotime Encoder
# ---------------------------------------------------------------------------

class PseudotimeEncoder(nn.Module):
    """
    Maps continuous pseudotime ∈ [0, 1] to d_model-dimensional embedding.
    Pseudotime is comparable across datasets via shared linear normalisation
    per dataset lineage tree.
    """

    def __init__(self, d_model: int = 256, n_bins: int = 50, dropout: float = 0.1):
        super().__init__()
        self.n_bins = n_bins

        # Soft-bin pseudotime and project
        self.bin_embed = nn.Embedding(n_bins + 1, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pseudotime: torch.Tensor) -> torch.Tensor:
        """
        pseudotime: [B]  float in [0, 1]
        Returns [B, d_model]
        """
        bins = (pseudotime * self.n_bins).long().clamp(0, self.n_bins)
        emb = self.bin_embed(bins)
        return self.norm(self.proj(emb))


# ---------------------------------------------------------------------------
# 3. Gene Expression Tokeniser (simplified)
# ---------------------------------------------------------------------------

class GeneExpressionTokeniser(nn.Module):
    """Rank-value + bin tokeniser for gene expression."""

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_bins: int = 10,
        max_seq: int = 512,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_bins = n_bins
        self.max_seq = max_seq

        self.gene_embed = nn.Embedding(n_genes + 1, d_model, padding_idx=n_genes)
        self.bin_embed  = nn.Embedding(n_bins + 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def discretise(self, expr: torch.Tensor) -> torch.Tensor:
        log_expr = torch.log1p(expr)
        max_val = log_expr.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        return (log_expr / max_val * self.n_bins).long().clamp(0, self.n_bins)

    def forward(
        self,
        expr: torch.Tensor,
        mask_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns gene tokens [B, seq, d_model] and ground-truth bins [B, seq].
        """
        seq = min(self.n_genes, self.max_seq)
        bins_gt = self.discretise(expr)[:, :seq]   # [B, seq]
        bins = bins_gt.clone()
        if mask_ids is not None:
            bins[:, mask_ids] = self.n_bins + 1   # mask token

        gene_ids = torch.arange(seq, device=expr.device).unsqueeze(0)
        tokens = self.gene_embed(gene_ids.expand(expr.size(0), -1)) + self.bin_embed(bins)
        return self.norm(tokens), bins_gt


# ---------------------------------------------------------------------------
# 4. Modality-Aware Transformer Block
# ---------------------------------------------------------------------------

class ModalityAwareBlock(nn.Module):
    """Standard transformer block with modality-conditioning via FiLM."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_modalities: int = 4,  # RNA, spatial, pseudotime, protein
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # FiLM conditioning: scale + shift from modality embedding
        self.mod_embed = nn.Embedding(n_modalities, d_model * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FiLM modality conditioning
        if modality_ids is not None:
            film = self.mod_embed(modality_ids)  # [B, 2*d]
            gamma, beta = film.chunk(2, dim=-1)  # each [B, d]
            x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ---------------------------------------------------------------------------
# 5. Full SpaceTime-scFM
# ---------------------------------------------------------------------------

class SpaceTimescFM(nn.Module):
    """
    Mosaic-friendly spatial-temporal single-cell foundation model.

    The model concatenates available modality tokens:
      [CLS] [cell_gene_tokens] [spatial_token (optional)] [pseudotime_token (optional)]

    Pretraining losses:
      a) Masked gene expression → cross-entropy on bins
      b) Masked spatial position → MSE on position embedding
      c) Pseudotime regression → MSE on pseudotime value
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        n_bins: int = 10,
        max_seq: int = 512,
        spatial_dim: int = 2,
        n_pt_bins: int = 50,
        n_modalities: int = 4,
        mask_ratio: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes  = n_genes
        self.n_bins   = n_bins
        self.mask_ratio = mask_ratio

        # Tokenisers
        self.gene_tok = GeneExpressionTokeniser(n_genes, d_model, n_bins, max_seq)
        self.spatial_enc = SpatialPositionEncoder(d_model, spatial_dim, dropout)
        self.pt_enc = PseudotimeEncoder(d_model, n_pt_bins, dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModalityAwareBlock(d_model, n_heads, n_modalities, dropout)
            for _ in range(n_layers)
        ])

        # Prediction heads
        self.gene_pred_head    = nn.Linear(d_model, n_bins + 1)     # bin recon
        self.spatial_pred_head = nn.Linear(d_model, d_model)         # spatial emb recon
        self.pt_pred_head      = nn.Linear(d_model, 1)               # pseudotime regression

        self.d_model = d_model

    def _random_mask_ids(self, seq_len: int, device) -> torch.Tensor:
        n_mask = max(1, int(seq_len * self.mask_ratio))
        perm = torch.randperm(seq_len, device=device)
        return perm[:n_mask].sort().values

    def forward(
        self,
        expr: torch.Tensor,                      # [B, n_genes]
        coords: Optional[torch.Tensor] = None,   # [B, spatial_dim]
        pseudotime: Optional[torch.Tensor] = None,  # [B]
        modality_ids: Optional[torch.Tensor] = None,  # [B]
        mask_genes: bool = True,
        mask_spatial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = expr.size(0)
        seq_len = min(self.n_genes, self.gene_tok.max_seq)
        device = expr.device

        # ---- Gene tokens ----
        mask_ids = self._random_mask_ids(seq_len, device) if mask_genes else None
        gene_tokens, bins_gt = self.gene_tok(expr, mask_ids=mask_ids)  # [B, seq, d]

        # ---- Spatial token ----
        spatial_token = None
        spatial_emb_gt = None
        if coords is not None:
            spatial_emb_gt = self.spatial_enc(coords).detach()  # target
            if mask_spatial:
                # Replace with learned mask token
                spatial_token = torch.zeros(B, 1, self.d_model, device=device)
            else:
                spatial_token = spatial_emb_gt.unsqueeze(1)     # [B, 1, d]

        # ---- Pseudotime token ----
        pt_token = None
        if pseudotime is not None:
            pt_token = self.pt_enc(pseudotime).unsqueeze(1)     # [B, 1, d]

        # ---- Concatenate all tokens ----
        cls = self.cls_token.expand(B, -1, -1)
        parts = [cls, gene_tokens]
        if spatial_token is not None:
            parts.append(spatial_token)
        if pt_token is not None:
            parts.append(pt_token)

        x = torch.cat(parts, dim=1)  # [B, total_len, d]

        for block in self.blocks:
            x = block(x, modality_ids=modality_ids)

        # ---- Losses ----
        losses = {}
        total_loss = torch.tensor(0.0, device=device)

        # (a) Masked gene reconstruction
        if mask_ids is not None:
            masked_repr = x[:, 1 + mask_ids, :]          # [B, n_mask, d]
            logits = self.gene_pred_head(masked_repr)     # [B, n_mask, n_bins+1]
            targets = bins_gt[:, mask_ids].long()
            gene_loss = F.cross_entropy(
                logits.reshape(-1, self.n_bins + 1),
                targets.reshape(-1),
            )
            losses["gene_loss"] = gene_loss
            total_loss = total_loss + gene_loss

        # (b) Spatial position reconstruction
        if coords is not None and mask_spatial:
            # Spatial token is the last appended token
            n_extra = (1 if spatial_token is not None else 0) + \
                      (1 if pt_token is not None else 0)
            spatial_repr = x[:, -n_extra, :]  # [B, d]
            spatial_pred = self.spatial_pred_head(spatial_repr)
            spatial_loss = F.mse_loss(spatial_pred, spatial_emb_gt)
            losses["spatial_loss"] = spatial_loss
            total_loss = total_loss + spatial_loss

        # (c) Pseudotime regression
        if pseudotime is not None:
            pt_repr = x[:, -1, :]            # [B, d]
            pt_pred = self.pt_pred_head(pt_repr).squeeze(-1)   # [B]
            pt_loss = F.mse_loss(pt_pred, pseudotime)
            losses["pt_loss"] = pt_loss
            total_loss = total_loss + pt_loss

        losses["loss"] = total_loss
        losses["cell_emb"] = x[:, 0, :]     # CLS token

        # ---- Spatial contrastive loss ----
        if coords is not None and not mask_spatial:
            pos_emb = self.spatial_enc(coords)
            sp_contrast = self.spatial_enc.contrastive_loss(coords, pos_emb)
            losses["spatial_contrast"] = sp_contrast
            losses["loss"] = losses["loss"] + 0.1 * sp_contrast

        return losses

    def get_cell_embeddings(
        self,
        expr: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        pseudotime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(expr, coords, pseudotime, mask_genes=False)
        return out["cell_emb"]
