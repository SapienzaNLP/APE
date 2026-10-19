"""
Objective definition for APE training.
"""

import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class APELoss(nn.Module):
    """
    APE objective function.
    """

    def __init__(
        self,
        emb_loss_weight: float = 1.0,
        act_loss_weight: float = 1.0,
        rep_loss_weight: float = 0.1,
        activity_threshold: float = 0.5,
    ):
        super().__init__()
        self.emb_loss_weight = emb_loss_weight
        self.act_loss_weight = act_loss_weight
        self.rep_loss_weight = rep_loss_weight
        self.activity_threshold = activity_threshold

    @staticmethod
    def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Pairwise cosine distance between rows of "a" [N_Slots_a, D_Model] and "b" [N_Slots_b, D_Model]
        """
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return 1.0 - (a_n @ b_n.T)

    @staticmethod
    @torch.no_grad()
    def hungarian_match(
        slot_embeds: torch.Tensor,
        tgt_embeds: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match N slots to K targets. Returns (Slot_idx, Target_idx) of length K.
        """
        cost = (
            APELoss.cosine_distance_matrix(slot_embeds, tgt_embeds)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        slot_idx, tgt_idx = linear_sum_assignment(cost)
        return slot_idx, tgt_idx

    def slot_repulsion_loss(
        self,
        slot_embeds: torch.Tensor,
        slot_act_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourages active slots to be dissimilar to each other.
        Penalizes pairs where at least one slot is active.
        """
        B, N, _ = slot_embeds.shape
        device = slot_embeds.device

        normed = F.normalize(slot_embeds, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))

        act_probs = torch.sigmoid(slot_act_logits)
        active = act_probs > self.activity_threshold

        # at least one active, off-diagonal
        pair_mask = active.unsqueeze(2) | active.unsqueeze(1)
        off_diag = ~torch.eye(N, dtype=torch.bool, device=device)
        final_mask = pair_mask & off_diag.unsqueeze(0)

        if not final_mask.any():
            return slot_embeds.sum() * 0.0

        return sim[final_mask].clamp(min=0).mean()

    def forward(
        self,
        slot_embeds: torch.Tensor,
        slot_act_logits: torch.Tensor,
        tgt_embeds: torch.Tensor,
        atom_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns (total_loss, act_tgts, stats).
        """
        B, N, _ = slot_embeds.shape
        device = slot_embeds.device

        emb_losses: List[torch.Tensor] = []
        act_tgts = torch.zeros(B, N, device=device)
        matched_count = 0

        for i in range(B):
            s, e = atom_offsets[i].item(), atom_offsets[i + 1].item()
            if e == s:
                continue

            tgts = tgt_embeds[s:e]
            slots_i = slot_embeds[i]

            slot_idx, tgt_idx = self.hungarian_match(slots_i, tgts)

            matched_slot_emb = slots_i[slot_idx]
            matched_tgt = tgts[tgt_idx]

            cos_sim = F.cosine_similarity(matched_slot_emb, matched_tgt, dim=-1)
            emb_losses.append((1.0 - cos_sim).sum())

            act_tgts[i, slot_idx] = 1.0
            matched_count += len(slot_idx)

        if matched_count == 0:
            emb_loss = slot_embeds.sum() * 0.0
        else:
            emb_loss = torch.stack(emb_losses).sum() / matched_count

        act_loss = F.binary_cross_entropy_with_logits(slot_act_logits, act_tgts)
        rep_loss = self.slot_repulsion_loss(slot_embeds, slot_act_logits)

        total = (
            self.emb_loss_weight * emb_loss
            + self.act_loss_weight * act_loss
            + self.rep_loss_weight * rep_loss
        )

        stats = {
            "loss_emb": emb_loss.detach().item(),
            "loss_act": act_loss.detach().item(),
            "loss_rep": rep_loss.detach().item(),
            "loss_total": total.detach().item(),
            "matched": matched_count,
        }

        return total, act_tgts, stats
