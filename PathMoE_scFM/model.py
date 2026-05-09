"""
PathMoE-scFM: Pathway-Aware Sparse Mixture-of-Experts Transformer
=================================================================
Replaces the monolithic feed-forward layers in a transformer with
sparse MoE layers where each expert corresponds to a curated biological
pathway (Reactome, Hallmark, KEGG, TF-regulon from DoRothEA).

Gaps addressed:
  B  – Biological priors are discarded
  D  – Poorly calibrated uncertainty (expert-disagreement signal)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Gene Tokeniser
# ---------------------------------------------------------------------------

class GeneTokeniser(nn.Module):
    """
    Maps a (cell, gene) count matrix to a sequence of gene tokens.

    Each input row is a sparse vector of shape [n_genes].
    We discretise expression into B bins per gene, then embed
    (gene_id, bin_id) pairs as a single dense vector.
    """

    def __init__(
        self,
        n_genes: int,
        n_bins: int = 10,
        d_model: int = 256,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_bins = n_bins
        self.d_model = d_model

        self.gene_embed = nn.Embedding(n_genes + 1, d_model, padding_idx=n_genes)
        self.bin_embed = nn.Embedding(n_bins + 2, d_model)  # +2 for zero / pad
        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)

        self.norm = nn.LayerNorm(d_model)

    def discretise(self, counts: torch.Tensor) -> torch.Tensor:
        """Log1p then quantile-bin into [1..n_bins]; 0 stays 0."""
        log_counts = torch.log1p(counts)
        # Bin via linear scale (simple but sufficient for pretraining)
        max_val = log_counts.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        bins = (log_counts / max_val * self.n_bins).long().clamp(0, self.n_bins)
        return bins  # [batch, n_genes]

    def forward(
        self,
        counts: torch.Tensor,        # [B, n_genes]  raw counts
        gene_ids: Optional[torch.Tensor] = None,  # [n_genes]  global gene ids
        mask_gene_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
          tokens  : [B, seq_len, d_model]
          pad_mask: [B, seq_len]  True where padded
        """
        batch_size = counts.size(0)

        # Default: genes in row order
        if gene_ids is None:
            gene_ids = torch.arange(self.n_genes, device=counts.device)

        bins = self.discretise(counts)           # [B, n_genes]

        # Keep only non-zero genes (and up to max_seq_len)
        # Build a fixed-length token sequence
        seq_len = min(self.n_genes, 2048)
        g_ids = gene_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)  # [B, seq]
        b_ids = bins[:, :seq_len]                                        # [B, seq]

        # Replace masked genes with a special mask token
        if mask_gene_ids is not None:
            b_ids = b_ids.clone()
            b_ids[:, mask_gene_ids] = self.n_bins + 1  # mask bin

        pos = torch.arange(seq_len, device=counts.device).unsqueeze(0)

        tokens = self.gene_embed(g_ids) + self.bin_embed(b_ids) + self.pos_embed(pos)
        tokens = self.norm(tokens)

        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=counts.device)
        return tokens, pad_mask


# ---------------------------------------------------------------------------
# 2. Pathway-Expert Feed-Forward Network
# ---------------------------------------------------------------------------

class PathwayExpert(nn.Module):
    """A single pathway expert: a small 2-layer MLP."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PathwayMoE(nn.Module):
    """
    Sparse Mixture-of-Experts where each expert corresponds to a
    biological pathway.

    Args
    ----
    n_experts    : total number of pathway experts (e.g., 300)
    top_k        : number of active experts per token (e.g., 4)
    d_model      : model dimension
    d_ff         : expert hidden dimension
    gene_pathway_matrix : np.ndarray [n_genes, n_experts]  {0,1}  binary
                  prior membership (rows = genes, cols = experts).
                  Used to initialise the gating weights.
    """

    def __init__(
        self,
        n_experts: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        gene_pathway_matrix: Optional[np.ndarray] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # One expert per pathway
        self.experts = nn.ModuleList(
            [PathwayExpert(d_model, d_ff, dropout) for _ in range(n_experts)]
        )

        # Gating network: linear layer from token -> expert logits
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Initialise gating with biological prior if available
        if gene_pathway_matrix is not None:
            self._init_gate_from_prior(gene_pathway_matrix, d_model)

        self.dropout = nn.Dropout(dropout)

    def _init_gate_from_prior(self, gpm: np.ndarray, d_model: int) -> None:
        """
        Soft initialisation: pathways that the gene belongs to get
        higher initial logit via PCA of the membership matrix.
        This is a heuristic; training will refine it.
        """
        gpm_t = torch.tensor(gpm, dtype=torch.float32)  # [n_genes, n_experts]
        # Use SVD to project pathway membership into d_model space
        if gpm_t.shape[0] >= d_model and gpm_t.shape[1] >= self.n_experts:
            U, S, Vt = torch.linalg.svd(gpm_t, full_matrices=False)
            k = min(d_model, S.shape[0])
            init_weight = (Vt[:k, :] * S[:k].unsqueeze(-1)).T  # [n_experts, k]
            # Pad or truncate to d_model
            if k < d_model:
                pad = torch.zeros(self.n_experts, d_model - k)
                init_weight = torch.cat([init_weight, pad], dim=-1)
            else:
                init_weight = init_weight[:, :d_model]
            with torch.no_grad():
                self.gate.weight.copy_(init_weight)

    def load_balance_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary load-balancing loss (Switch Transformer style).
        Encourages uniform expert utilisation.
        gate_logits: [B, seq, n_experts]
        """
        # fraction of tokens dispatched to each expert
        probs = F.softmax(gate_logits, dim=-1)  # [B, seq, n_experts]
        mean_prob = probs.mean(dim=[0, 1])       # [n_experts]
        # fraction of top-k selections going to each expert
        _, top_ids = torch.topk(gate_logits, self.top_k, dim=-1)
        dispatch = torch.zeros_like(mean_prob)
        for eid in range(self.n_experts):
            dispatch[eid] = (top_ids == eid).float().mean()
        loss = self.n_experts * (mean_prob * dispatch).sum()
        return loss

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : [B, seq, d_model]
        Returns (output [B, seq, d_model], load_balance_loss scalar)
        """
        B, S, D = x.shape
        gate_logits = self.gate(x)                       # [B, S, n_experts]
        lb_loss = self.load_balance_loss(gate_logits)

        # Hard top-k routing
        top_vals, top_ids = torch.topk(gate_logits, self.top_k, dim=-1)
        # Softmax over selected experts
        top_weights = F.softmax(top_vals, dim=-1)        # [B, S, k]

        # Compute expert outputs
        output = torch.zeros_like(x)
        for ki in range(self.top_k):
            expert_ids = top_ids[:, :, ki]               # [B, S]
            weights = top_weights[:, :, ki].unsqueeze(-1)  # [B, S, 1]
            # Batch all tokens for the same expert together
            for eid in range(self.n_experts):
                mask = (expert_ids == eid)               # [B, S]
                if mask.any():
                    tok_in = x[mask]                     # [n_tok, D]
                    tok_out = self.experts[eid](tok_in)  # [n_tok, D]
                    # weight by gate
                    w = weights[mask]                    # [n_tok, 1]
                    output[mask] = output[mask] + tok_out * w

        return output, lb_loss


# ---------------------------------------------------------------------------
# 3. Transformer Block with PathwayMoE
# ---------------------------------------------------------------------------

class PathMoETransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        d_ff: int,
        gene_pathway_matrix: Optional[np.ndarray] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = PathwayMoE(
            n_experts, top_k, d_model, d_ff,
            gene_pathway_matrix=gene_pathway_matrix,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # PathwayMoE FFN
        moe_out, lb_loss = self.moe(x)
        x = self.norm2(x + self.dropout(moe_out))

        return x, lb_loss


# ---------------------------------------------------------------------------
# 4. Full PathMoE-scFM
# ---------------------------------------------------------------------------

class PathMoEscFM(nn.Module):
    """
    Full PathMoE-scFM architecture.

    Pretraining objective: masked gene expression prediction.
    The model receives a cell with some gene bins masked and must
    reconstruct the original bin labels.
    """

    def __init__(
        self,
        n_genes: int,
        n_bins: int = 10,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        n_experts: int = 50,
        top_k: int = 4,
        d_ff: int = 512,
        gene_pathway_matrix: Optional[np.ndarray] = None,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.n_bins = n_bins
        self.n_genes = n_genes

        self.tokeniser = GeneTokeniser(n_genes, n_bins, d_model)
        self.blocks = nn.ModuleList([
            PathMoETransformerBlock(
                d_model, n_heads, n_experts, top_k, d_ff,
                gene_pathway_matrix=gene_pathway_matrix,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        # Prediction head: predict bin class for each masked gene
        self.pred_head = nn.Linear(d_model, n_bins + 1)

        # CLS token for cell-level embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _random_mask(self, n_seq: int, device) -> torch.Tensor:
        """Returns sorted indices of masked positions."""
        n_mask = max(1, int(n_seq * self.mask_ratio))
        perm = torch.randperm(n_seq, device=device)
        return perm[:n_mask].sort().values

    def encode(
        self,
        counts: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass without masking; returns [B, seq+1, d_model]."""
        tokens, pad_mask = self.tokeniser(counts, gene_ids)
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        total_lb = 0.0
        for block in self.blocks:
            x, lb = block(x)
            total_lb = total_lb + lb

        return x, total_lb

    def forward(
        self,
        counts: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward: random masking + reconstruction.

        Returns a dict with keys: 'loss', 'lb_loss', 'logits', 'mask_ids'.
        """
        tokens, pad_mask = self.tokeniser(counts, gene_ids)
        seq_len = tokens.size(1)

        # Ground-truth bins (before masking)
        bins_gt = self.tokeniser.discretise(counts)[:, :seq_len]  # [B, seq]

        # Mask random genes
        mask_ids = self._random_mask(seq_len, counts.device)
        tokens_masked, _ = self.tokeniser(
            counts, gene_ids, mask_gene_ids=mask_ids
        )

        B = tokens_masked.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens_masked], dim=1)

        total_lb = 0.0
        for block in self.blocks:
            x, lb = block(x)
            total_lb = total_lb + lb

        # Prediction on masked positions only (offset by 1 for CLS)
        masked_repr = x[:, 1 + mask_ids, :]          # [B, n_mask, d]
        logits = self.pred_head(masked_repr)           # [B, n_mask, n_bins+1]

        targets = bins_gt[:, mask_ids].long()          # [B, n_mask]
        recon_loss = F.cross_entropy(
            logits.reshape(-1, self.n_bins + 1),
            targets.reshape(-1),
        )

        loss = recon_loss + 0.01 * total_lb  # λ_lb = 0.01

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "lb_loss": total_lb,
            "logits": logits,
            "mask_ids": mask_ids,
            "cell_emb": x[:, 0, :],  # CLS token
        }

    def get_cell_embeddings(self, counts: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embeddings for downstream tasks."""
        with torch.no_grad():
            x, _ = self.encode(counts)
        return x[:, 0, :]


# ---------------------------------------------------------------------------
# 5. Fine-tuning head: cell-type annotation
# ---------------------------------------------------------------------------

class CellTypeClassifier(nn.Module):
    """Attach to PathMoEscFM for cell-type fine-tuning."""

    def __init__(self, backbone: PathMoEscFM, n_classes: int, d_model: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        cell_emb = self.backbone.get_cell_embeddings(counts)
        return self.head(cell_emb)
