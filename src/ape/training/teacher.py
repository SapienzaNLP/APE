"""
Utilities for teacher encoder.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load_frozen_teacher(teacher_name: str, device):
    teacher = AutoModel.from_pretrained(teacher_name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    return teacher, tokenizer


def pool(
    token_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "cls",   # "cls" or "mean"
) -> torch.Tensor:
    """
    Standard [CLS] | mean pooling with attention mask.
    """
    if mode == "cls":
        return token_states[:, 0, :]

    elif mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(token_states.dtype)
        summed = (token_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    raise ValueError(f"Unknown pool mode '{mode}', expected 'cls' or 'mean'")


@torch.no_grad()
def encode_gt_atoms(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_mode: str = "cls",
) -> torch.Tensor:
    """
    Pooled atom embeddings from the frozen teacher.
    """
    out = teacher(input_ids=input_ids, attention_mask=attention_mask)
    pooled = pool(out.last_hidden_state, attention_mask, mode=pool_mode)
    return pooled
