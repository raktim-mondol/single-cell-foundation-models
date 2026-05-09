"""
Atlas-Streamer: Continual Learning for Single-Cell Foundation Models
=====================================================================
Maintains a deployed scFM and updates it incrementally as new atlas
data arrives, WITHOUT catastrophic forgetting.

Core components:
  1. Frozen teacher (Mₜ) + student (Mₜ₊₁) self-distillation
  2. Importance-weighted experience replay buffer
     (over-samples rare cell types and minority demographic donors)
  3. Gene-vocabulary expansion via kNN over gene-embedding space

Gaps addressed:
  E  – Static models in a streaming-data world
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Simple scFM Backbone (teacher / student share this architecture)
# ---------------------------------------------------------------------------

class SimpleScFMBackbone(nn.Module):
    """
    Lightweight transformer backbone for use as teacher/student.
    In real deployment this would be replaced with Geneformer / scGPT.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_bins: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_bins = n_bins
        self.d_model = d_model

        self.gene_embed = nn.Embedding(n_genes + 1, d_model, padding_idx=n_genes)
        self.bin_embed  = nn.Embedding(n_bins + 2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pred_head = nn.Linear(d_model, n_bins + 1)  # masked-gene prediction

    def discretise(self, counts: torch.Tensor) -> torch.Tensor:
        log_c = torch.log1p(counts)
        mx = log_c.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        return (log_c / mx * self.n_bins).long().clamp(0, self.n_bins)

    def embed(self, counts: torch.Tensor, mask_ids: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (sequence output [B, 1+seq, d_model], bins_gt [B, seq]).
        """
        B = counts.size(0)
        seq = min(self.n_genes, 512)
        bins = self.discretise(counts)[:, :seq]
        bins_in = bins.clone()
        if mask_ids is not None:
            bins_in[:, mask_ids] = self.n_bins + 1

        gene_ids = torch.arange(seq, device=counts.device).unsqueeze(0).expand(B, -1)
        x = self.gene_embed(gene_ids) + self.bin_embed(bins_in)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return x, bins

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Returns CLS-token cell embeddings [B, d_model]."""
        x, _ = self.embed(counts)
        return x[:, 0, :]

    def masked_forward(self, counts: torch.Tensor, mask_ids: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, n_mask, n_bins+1], bins_gt [B, seq])."""
        x, bins_gt = self.embed(counts, mask_ids=mask_ids)
        masked = x[:, 1 + mask_ids, :]
        logits = self.pred_head(masked)
        return logits, bins_gt


# ---------------------------------------------------------------------------
# 2. Importance-Weighted Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular buffer with importance-weighted sampling.

    Importance weights are higher for:
      - Rare cell types (cell_type_rarity ∝ 1/freq)
      - Minority demographic donors (demographic_weight)
    """

    def __init__(self, capacity: int, n_genes: int):
        self.capacity = capacity
        self.n_genes = n_genes

        self.counts         = np.zeros((capacity, n_genes), dtype=np.float32)
        self.cell_type_ids  = np.zeros(capacity, dtype=np.int64)
        self.donor_ids      = np.zeros(capacity, dtype=np.int64)
        self.importance     = np.ones(capacity, dtype=np.float32)

        self._ptr   = 0
        self._size  = 0

    def add(
        self,
        counts: np.ndarray,          # [B, n_genes] — may be fewer cols than buffer
        cell_type_ids: np.ndarray,   # [B]
        donor_ids: np.ndarray,       # [B]
        importance: np.ndarray,      # [B]
    ):
        B = counts.shape[0]
        for i in range(B):
            idx = self._ptr % self.capacity
            col_end = min(counts.shape[1], self.n_genes)
            self.counts[idx, :col_end] = counts[i, :col_end]
            self.cell_type_ids[idx]    = cell_type_ids[i]
            self.donor_ids[idx]        = donor_ids[i]
            self.importance[idx]       = importance[i]
            self._ptr += 1
        self._size = min(self._size + B, self.capacity)

    def sample(self, n: int, device) -> Dict[str, torch.Tensor]:
        """Sample n cells with probability ∝ importance weight."""
        if self._size == 0:
            return {}
        w = self.importance[:self._size]
        w = w / (w.sum() + 1e-8)
        idxs = np.random.choice(self._size, size=min(n, self._size),
                                replace=False, p=w)
        return {
            "counts":         torch.tensor(self.counts[idxs], device=device),
            "cell_type_ids":  torch.tensor(self.cell_type_ids[idxs], device=device),
            "donor_ids":      torch.tensor(self.donor_ids[idxs], device=device),
        }

    def compute_importance(
        self,
        cell_type_ids: np.ndarray,
        donor_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Importance = (cell-type rarity) × (donor demographic weight).
        Rarity = 1 / freq(cell_type).
        Demographic weight = 1 / freq(donor) for minority donors.
        """
        ct_counts = np.bincount(cell_type_ids, minlength=cell_type_ids.max() + 1)
        donor_counts = np.bincount(donor_ids, minlength=donor_ids.max() + 1)

        ct_rarity = 1.0 / (ct_counts[cell_type_ids] + 1)
        d_weight  = 1.0 / (donor_counts[donor_ids] + 1)

        importance = ct_rarity * d_weight
        importance = importance / (importance.mean() + 1e-8)  # normalise
        return importance.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Gene Vocabulary Expansion
# ---------------------------------------------------------------------------

class GeneVocabExpander:
    """
    When new datasets introduce gene IDs not in the existing vocabulary,
    initialise their embeddings via kNN over sequence-similarity space.

    In practice, gene similarity is derived from:
      - Ensembl orthologue tables
      - Gene2Vec or protein sequence embeddings
    For the demo, we use random cosine-similarity neighbours.
    """

    @staticmethod
    def expand_embedding(
        existing_embed: nn.Embedding,
        new_gene_ids: List[int],
        similarity_matrix: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> nn.Embedding:
        """
        Returns a new Embedding module with extra rows for new genes.

        existing_embed : Embedding(n_old, d_model)
        new_gene_ids   : list of new gene indices to add
        similarity_matrix : [n_new, n_old] cosine similarity to old genes
        """
        n_old, d_model = existing_embed.weight.shape
        n_new = len(new_gene_ids)

        new_embed = nn.Embedding(n_old + n_new, d_model,
                                  padding_idx=existing_embed.padding_idx)

        with torch.no_grad():
            new_embed.weight[:n_old].copy_(existing_embed.weight)

            for i, gid in enumerate(new_gene_ids):
                if similarity_matrix is not None:
                    # kNN average of most similar existing genes
                    sim_row = similarity_matrix[i]
                    top_k = np.argsort(-sim_row)[:k]
                    weights = torch.tensor(sim_row[top_k], dtype=torch.float32)
                    weights = F.softmax(weights, dim=0)
                    knn_emb = (
                        existing_embed.weight[top_k].detach()
                        * weights.unsqueeze(-1)
                    ).sum(0)
                    new_embed.weight[n_old + i].copy_(knn_emb)
                else:
                    # Fallback: small random initialisation
                    nn.init.trunc_normal_(new_embed.weight[n_old + i : n_old + i + 1],
                                         std=0.02)

        return new_embed


# ---------------------------------------------------------------------------
# 4. Atlas-Streamer: Self-Distillation + Replay Update
# ---------------------------------------------------------------------------

class AtlasStreamer:
    """
    Manages the continual-learning update cycle for an scFM.

    Protocol for each new data release:
      1. Snapshot current model as frozen teacher Mₜ
      2. Fine-tune student Mₜ₊₁ on (new data + replay buffer)
         with masked-gene loss + KL distillation
      3. Update replay buffer with new data (importance-weighted)
      4. Expand gene vocabulary if new genes are present
      5. Swap: student becomes new teacher
    """

    def __init__(
        self,
        model: SimpleScFMBackbone,
        buffer_capacity: int = 50_000,
        distill_weight: float = 0.5,    # KL distillation coefficient
        mask_ratio: float = 0.15,
    ):
        self.student = model
        self.teacher: Optional[SimpleScFMBackbone] = None
        self.replay  = ReplayBuffer(buffer_capacity, model.n_genes)
        self.distill_weight = distill_weight
        self.mask_ratio = mask_ratio

    def snapshot_teacher(self):
        """Freeze current student as teacher."""
        import copy
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _random_mask(self, seq_len: int, device) -> torch.Tensor:
        n_mask = max(1, int(seq_len * self.mask_ratio))
        return torch.randperm(seq_len, device=device)[:n_mask].sort().values

    def update(
        self,
        new_counts: np.ndarray,         # [B_new, n_genes_new]
        new_cell_types: np.ndarray,     # [B_new]
        new_donor_ids: np.ndarray,      # [B_new]
        n_steps: int = 500,
        lr: float = 1e-4,
        replay_batch: int = 128,
        new_batch: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        One continual-learning update cycle.

        Returns dict of training statistics.
        """
        # ---- Expand vocabulary if needed ----
        n_new_genes = new_counts.shape[1]
        if n_new_genes > self.student.n_genes:
            new_ids = list(range(self.student.n_genes, n_new_genes))
            self.student.gene_embed = GeneVocabExpander.expand_embedding(
                self.student.gene_embed, new_ids
            )
            self.student.n_genes = n_new_genes
            # Expand replay buffer too
            new_buf = ReplayBuffer(self.replay.capacity, n_new_genes)
            new_buf.counts[:, :self.replay.n_genes] = self.replay.counts
            new_buf.cell_type_ids = self.replay.cell_type_ids
            new_buf.donor_ids     = self.replay.donor_ids
            new_buf.importance    = self.replay.importance
            new_buf._ptr  = self.replay._ptr
            new_buf._size = self.replay._size
            self.replay = new_buf

        # ---- Snapshot teacher ----
        self.snapshot_teacher()
        self.student.to(device)
        self.teacher.to(device)

        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=1e-4
        )

        stats = {"recon_loss": 0.0, "kl_loss": 0.0, "n_steps": n_steps}
        new_counts_f = torch.tensor(new_counts, dtype=torch.float32)

        for step in range(n_steps):
            # Sample new data mini-batch
            idxs = np.random.choice(len(new_counts), size=min(new_batch, len(new_counts)),
                                    replace=False)
            new_batch_c = new_counts_f[idxs].to(device)

            # Sample replay
            replay_batch_d = self.replay.sample(replay_batch, device)
            if "counts" in replay_batch_d:
                all_counts = torch.cat([new_batch_c, replay_batch_d["counts"]], dim=0)
            else:
                all_counts = new_batch_c

            # Align gene dimension
            if all_counts.shape[1] < self.student.n_genes:
                pad = torch.zeros(
                    all_counts.shape[0],
                    self.student.n_genes - all_counts.shape[1],
                    device=device,
                )
                all_counts = torch.cat([all_counts, pad], dim=1)

            seq_len = min(self.student.n_genes, 512)
            mask_ids = self._random_mask(seq_len, device)

            # Student forward (masked gene prediction)
            student_logits, bins_gt = self.student.masked_forward(all_counts, mask_ids)
            targets = bins_gt[:, mask_ids].long()
            recon_loss = F.cross_entropy(
                student_logits.reshape(-1, self.student.n_bins + 1),
                targets.reshape(-1),
            )

            # KL distillation on replay (keep old knowledge)
            kl_loss = torch.tensor(0.0, device=device)
            if "counts" in replay_batch_d:
                rp = replay_batch_d["counts"]
                if rp.shape[1] < self.student.n_genes:
                    rp = torch.cat([
                        rp,
                        torch.zeros(rp.shape[0],
                                    self.student.n_genes - rp.shape[1],
                                    device=device)
                    ], dim=1)
                with torch.no_grad():
                    teacher_emb = self.teacher(rp)
                student_emb = self.student(rp)
                # KL between unit-Gaussian-approximated distributions
                kl_loss = F.mse_loss(student_emb, teacher_emb)

            loss = recon_loss + self.distill_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()

            stats["recon_loss"] += recon_loss.item()
            stats["kl_loss"]    += kl_loss.item()

        stats["recon_loss"] /= n_steps
        stats["kl_loss"]    /= n_steps

        # ---- Update replay buffer ----
        importance = self.replay.compute_importance(new_cell_types, new_donor_ids)
        self.replay.add(new_counts, new_cell_types, new_donor_ids, importance)

        return stats

    def backward_transfer(
        self,
        old_counts: torch.Tensor,
        old_labels: torch.Tensor,
        classifier: nn.Module,
        device: torch.device,
    ) -> float:
        """
        Approximate backward transfer: classification accuracy on old data
        after update. Should remain within 2% of pre-update accuracy.
        """
        self.student.eval()
        with torch.no_grad():
            embs = self.student(old_counts.to(device))
            preds = classifier(embs).argmax(dim=-1)
        acc = (preds.cpu() == old_labels).float().mean().item()
        return acc
