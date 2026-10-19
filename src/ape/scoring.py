"""
Scoring rules for pairs of APE proposition sets.

Convention for asymmetric rules (h_*):
    emb_a = hypothesis (query), emb_b = premise (document).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class APEScore:
    """
    Caches the normalized embeddings and similarity matrix for
    a sentence pair. Then, defines scoring rules as methods.
    """

    RULES = ["hungarian", "maxsim", "h_min_to_p", "pool"]

    def __init__(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> None:
        self.emb_a = F.normalize(torch.as_tensor(emb_a).float(), dim=-1)
        self.emb_b = F.normalize(torch.as_tensor(emb_b).float(), dim=-1)
        self.sim = self.emb_a @ self.emb_b.T

    def _empty(self) -> bool:
        return self.emb_a.numel() == 0 or self.emb_b.numel() == 0

    def similarity_matrix(self) -> np.ndarray:
        """
        Cosine similarity matrix between A-atoms (rows) and B-atoms (columns).
        """
        return self.sim.detach().cpu().numpy()

    def print_similarity_matrix(self, precision: int = 3) -> None:
        """
        Pretty-print the similarity matrix, rows=A-atoms, columns=B-atoms.
        """
        sim = self.similarity_matrix()
        if sim.size == 0:
            print("(empty similarity matrix)")
            return

        col_labels = [f"B{j}" for j in range(sim.shape[1])]
        row_label_width = max(len(f"A{i}") for i in range(sim.shape[0])) + 1
        col_width = max(precision + 4, max(len(c) for c in col_labels) + 2)

        header = " " * row_label_width + "".join(c.rjust(col_width) for c in col_labels)
        print(header)
        for i, row in enumerate(sim):
            values = "".join(f"{v:.{precision}f}".rjust(col_width) for v in row)
            print(f"A{i}".ljust(row_label_width) + values)

    def hungarian(self) -> float:
        """
        Optimal one-to-one assignment. Averaged over matched pairs.
        """
        if self._empty():
            return 0.0
        sim = self.similarity_matrix()
        row, col = linear_sum_assignment(-sim)
        return float(sim[row, col].mean())

    def maxsim(self) -> float:
        """
        Symmetric mean of per-atom max cosines (both directions averaged).
        """
        if self._empty():
            return 0.0
        fwd = self.sim.max(dim=1).values.mean().item()
        bwd = self.sim.max(dim=0).values.mean().item()
        return 0.5 * (fwd + bwd)

    def pool(self) -> float:
        """
        Mean-pool each atom set to one vector. Then, cosine similarity.
        """
        if self._empty():
            return 0.0
        a = F.normalize(self.emb_a.mean(dim=0, keepdim=True), dim=-1)
        b = F.normalize(self.emb_b.mean(dim=0, keepdim=True), dim=-1)
        return float((a @ b.T).item())

    def h_min_to_p(self) -> float:
        """
        For each H-atom, take its max cosine to any P-atom and return the MIN.
        The weakest H-atom's best P-match functions as the bottleneck.
        """
        if self._empty():
            return 0.0
        per_h_max = self.sim.max(dim=1).values
        return float(per_h_max.min().item())

    def score(self, rule: str) -> float:
        if rule not in self.RULES:
            raise ValueError(f"Unknown scoring rule '{rule}'. Available: {self.RULES}")
        return getattr(self, rule)()
