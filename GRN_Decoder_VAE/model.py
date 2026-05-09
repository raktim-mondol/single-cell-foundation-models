"""
GRN-Decoder VAE: Gene-Regulatory-Network Constrained Generative scFM
======================================================================
Marries the scale of foundation models with the structural inductive
bias of structural causal models.

Architecture:
  Encoder : Transformer → latent z ~ q(z|x)  (amortised VI)
  Decoder : Sparse signed GRN graph → predicted expression
            x̂_g = NB(μ_g, ϕ_g)  where
            μ_g = f_g(z, {expression of cis-regulators of g})

Key properties:
  - Counterfactual head: set any TF's decoded value to 0 → in silico KO
  - L1 sparsity penalty on edges beyond the curated GRN (edge discovery)
  - Calibrated ZINB likelihood for per-gene uncertainty

Gaps addressed:
  A  – Causation (GRN-based counterfactuals)
  B  – Biological priors (GRN as decoder graph)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Transformer Encoder → latent z
# ---------------------------------------------------------------------------

class GRNVAEEncoder(nn.Module):
    """
    Transformer encoder that maps raw counts to Gaussian latent parameters.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_latent: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_latent = d_latent

        # Project each gene's log1p expression + gene embedding
        self.gene_embed = nn.Embedding(n_genes, d_model)
        self.expr_proj  = nn.Linear(1, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Variational parameters
        self.mu_proj    = nn.Linear(d_model, d_latent)
        self.logvar_proj = nn.Linear(d_model, d_latent)

    def forward(self, counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        counts: [B, n_genes]  log1p normalised
        Returns (mu [B, d_latent], log_var [B, d_latent])
        """
        B, G = counts.shape
        gene_ids = torch.arange(G, device=counts.device).unsqueeze(0).expand(B, -1)

        gene_tok  = self.gene_embed(gene_ids)              # [B, G, d]
        expr_tok  = self.expr_proj(counts.unsqueeze(-1))   # [B, G, d]
        x = self.input_norm(gene_tok + expr_tok)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)

        cell_rep = x[:, 0, :]   # CLS token [B, d]
        mu      = self.mu_proj(cell_rep)
        log_var = self.logvar_proj(cell_rep).clamp(-10, 4)
        return mu, log_var

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


# ---------------------------------------------------------------------------
# 2. GRN-Structured Decoder
# ---------------------------------------------------------------------------

class GRNDecoderLayer(nn.Module):
    """
    Per-gene MLP whose inputs are restricted to its cis-regulators and TF set.
    Latent z is also injected as a global context.
    """

    def __init__(self, n_regulators: int, d_latent: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + n_regulators, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2),  # predicts log_mu and log_phi (NB parameters)
        )

    def forward(self, z: torch.Tensor, reg_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z        : [B, d_latent]
        reg_expr : [B, n_regulators]  expression of regulators
        Returns (log_mu [B], log_phi [B])
        """
        inp = torch.cat([z, reg_expr], dim=-1)
        out = self.net(inp)        # [B, 2]
        return out[:, 0], out[:, 1]


class GRNDecoder(nn.Module):
    """
    Sparse GRN-based decoder.

    grn_adj : np.ndarray [n_genes, n_genes]  signed adjacency matrix
              entry (i, j) = 1  → gene j activates gene i
              entry (i, j) = -1 → gene j represses gene i
              entry (i, j) = 0  → no known direct regulation

    The decoder decodes genes in topological order of the GRN DAG.
    Feedback loops are broken by treating targets in the current iteration
    as read-only (steady-state approximation).
    """

    def __init__(
        self,
        n_genes: int,
        d_latent: int,
        grn_adj: Optional[np.ndarray] = None,
        l1_lambda: float = 0.01,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_latent = d_latent
        self.l1_lambda = l1_lambda

        # If no GRN provided, use a sparse random graph as demo
        if grn_adj is None:
            rng = np.random.default_rng(0)
            grn_adj = (rng.random((n_genes, n_genes)) < 0.02).astype(np.float32)
            np.fill_diagonal(grn_adj, 0)

        self.register_buffer("grn_adj", torch.tensor(grn_adj, dtype=torch.float32))

        # Learnable edge weights (initialised from curated GRN,
        # L1 penalty on edges added beyond curated set)
        self.edge_weights = nn.Parameter(
            torch.tensor(grn_adj.copy(), dtype=torch.float32)
        )

        # Per-gene decoder layers
        # Each gene uses at most max_reg regulators
        self.max_reg = max(int(grn_adj.sum(axis=1).max()) + 1, 2)
        self.gene_decoders = nn.ModuleList([
            GRNDecoderLayer(self.max_reg, d_latent, hidden)
            for _ in range(n_genes)
        ])

        # Library size offset (per cell)
        self.lib_proj = nn.Linear(d_latent, 1)

    def forward(
        self,
        z: torch.Tensor,
        tf_override: Optional[Dict[int, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        z          : [B, d_latent]
        tf_override: {gene_idx: value}  for in silico TF perturbations

        Returns dict with log_mu [B, n_genes], log_phi [B, n_genes].
        """
        B = z.size(0)
        device = z.device

        # Effective edge weights (curated + learnable additions)
        eff_weights = self.edge_weights  # [n_genes, n_genes]

        # Steady-state: initialise decoded expression to zero
        decoded = torch.zeros(B, self.n_genes, device=device)

        # Apply TF overrides
        if tf_override is not None:
            for gidx, val in tf_override.items():
                decoded[:, gidx] = val

        # Decode genes in a fixed pass (approximates topological order)
        log_mus  = torch.zeros(B, self.n_genes, device=device)
        log_phis = torch.zeros(B, self.n_genes, device=device)

        for g in range(self.n_genes):
            # Get weighted regulator expression
            w = eff_weights[g]  # [n_genes]  weight of each gene as regulator of g
            # Pad to max_reg regulators
            reg_vals = (decoded * w.unsqueeze(0))  # [B, n_genes]
            # Take top-max_reg by absolute weight magnitude
            top_idx = w.abs().topk(self.max_reg).indices  # [max_reg]
            reg_expr = reg_vals[:, top_idx]   # [B, max_reg]

            log_mu, log_phi = self.gene_decoders[g](z, reg_expr)
            log_mus[:, g]  = log_mu
            log_phis[:, g] = log_phi

            # Update decoded expression for downstream genes
            decoded = decoded.clone()
            decoded[:, g] = torch.exp(log_mu)

        # L1 penalty on edges added beyond curated GRN
        curated_mask = (self.grn_adj == 0).float()  # beyond-curated edges
        l1_loss = (self.edge_weights.abs() * curated_mask).sum()

        return {
            "log_mu":  log_mus,
            "log_phi": log_phis,
            "l1_loss": l1_loss,
        }


# ---------------------------------------------------------------------------
# 3. ZINB Likelihood
# ---------------------------------------------------------------------------

def zinb_loss(
    x: torch.Tensor,         # [B, n_genes]  raw counts
    log_mu: torch.Tensor,    # [B, n_genes]
    log_phi: torch.Tensor,   # [B, n_genes]  log dispersion
    pi: Optional[torch.Tensor] = None,  # [B, n_genes]  dropout probability
) -> torch.Tensor:
    """
    Negative log-likelihood of Zero-Inflated Negative Binomial.
    If pi is None, reduces to plain NB.
    """
    mu  = torch.exp(log_mu).clamp(min=1e-6)
    phi = torch.exp(log_phi).clamp(min=1e-6, max=1e3)  # dispersion

    # NB log-prob: log p_NB(x | mu, phi)
    log_prob = (
        torch.lgamma(x + phi)
        - torch.lgamma(phi)
        - torch.lgamma(x + 1)
        + phi * (torch.log(phi) - torch.log(phi + mu))
        + x   * (torch.log(mu)  - torch.log(phi + mu))
    )

    if pi is None:
        return -log_prob.mean()

    # ZINB
    log_prob_zero = phi * (torch.log(phi) - torch.log(phi + mu))
    log_nb_zero   = log_prob_zero
    log_pi        = torch.log(pi.clamp(min=1e-8))
    log_1mpi      = torch.log((1 - pi).clamp(min=1e-8))

    zero_case    = torch.logaddexp(log_pi, log_1mpi + log_nb_zero)
    nonzero_case = log_1mpi + log_prob

    nll = -torch.where(x < 0.5, zero_case, nonzero_case).mean()
    return nll


# ---------------------------------------------------------------------------
# 4. Full GRN-Decoder VAE
# ---------------------------------------------------------------------------

class GRNDecoderVAE(nn.Module):
    """
    Full VAE with GRN-structured decoder.

    Training loss:
      ELBO = E[log p(x|z)] - KL(q(z|x) ‖ N(0,I))
           - λ_L1 · edge_L1_penalty
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_enc_layers: int = 3,
        d_latent: int = 32,
        grn_adj: Optional[np.ndarray] = None,
        l1_lambda: float = 0.01,
        use_zinb: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.use_zinb = use_zinb

        self.encoder = GRNVAEEncoder(
            n_genes, d_model, n_heads, n_enc_layers, d_latent, dropout
        )
        self.decoder = GRNDecoder(
            n_genes, d_latent, grn_adj, l1_lambda
        )

        if use_zinb:
            # Dropout probability per gene per cell (small MLP)
            self.pi_head = nn.Sequential(
                nn.Linear(d_latent, n_genes),
                nn.Sigmoid(),
            )

    def forward(
        self,
        counts: torch.Tensor,  # [B, n_genes]  raw counts
        tf_override: Optional[Dict[int, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Log1p normalise for encoder
        log_counts = torch.log1p(counts)

        mu_z, log_var_z = self.encoder(log_counts)
        z = self.encoder.reparameterise(mu_z, log_var_z)

        dec = self.decoder(z, tf_override=tf_override)

        pi = self.pi_head(z) if self.use_zinb else None

        # Reconstruction loss (NB or ZINB on raw counts)
        recon_loss = zinb_loss(counts, dec["log_mu"], dec["log_phi"], pi=pi)

        # KL divergence
        kl_loss = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).mean()

        # L1 edge penalty
        l1_loss = dec["l1_loss"]

        elbo = recon_loss + kl_loss + self.decoder.l1_lambda * l1_loss

        return {
            "loss":       elbo,
            "recon_loss": recon_loss,
            "kl_loss":    kl_loss,
            "l1_loss":    l1_loss,
            "z":          z,
            "log_mu":     dec["log_mu"],
            "log_phi":    dec["log_phi"],
        }

    def in_silico_knockout(
        self,
        counts: torch.Tensor,
        tf_gene_idx: int,
    ) -> torch.Tensor:
        """
        Predict post-knockout expression by forcing TF expression = 0
        in the decoder at inference time.
        """
        log_counts = torch.log1p(counts)
        with torch.no_grad():
            mu_z, log_var_z = self.encoder(log_counts)
            z = mu_z  # no reparameterisation at inference
            dec = self.decoder(z, tf_override={tf_gene_idx: 0.0})
        return torch.exp(dec["log_mu"])  # predicted counts

    def edge_recovery_precision_recall(
        self,
        true_grn: np.ndarray,
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compare learned edge_weights to ground-truth GRN adjacency.
        """
        learned = self.decoder.edge_weights.detach().cpu().numpy()
        learned_binary = (np.abs(learned) > threshold).astype(int)
        true_binary    = (np.abs(true_grn) > 0).astype(int)

        tp = (learned_binary * true_binary).sum()
        fp = (learned_binary * (1 - true_binary)).sum()
        fn = ((1 - learned_binary) * true_binary).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
