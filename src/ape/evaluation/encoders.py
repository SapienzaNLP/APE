"""
Sentence encoders compared in the benchmarks: APE (multi-vector) and pooled fixed-size encoders
(APE's teacher, the fixed-size baseline). Each maps sentences to L2-normalized embedding sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ape.model import APE
from ape.training.teacher import pool


@dataclass
class APEEncoder:
    """
    APE: each sentence -> [n_propositions, D]. Scored with every `APEScore` rule.
    """
    model: APE
    activity_threshold: Optional[float] = None
    batch_size: int = 128
    max_seq_len: int = 128
    name: str = "ape"
    multi_vector: bool = True

    def encode(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        unique = list(dict.fromkeys(sentences))
        embeds = self.model.encode(
            unique,
            batch_size=self.batch_size,
            activity_threshold=self.activity_threshold,
            max_seq_len=self.max_seq_len,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return dict(zip(unique, embeds))


@dataclass
class PooledEncoder:
    """
    Fixed-size encoder: each sentence -> [1, D] ("cls" or "mean" pooling). Scored with cosine.
    """
    name: str
    model: nn.Module
    tokenizer: Any
    pool_mode: str = "mean"
    batch_size: int = 128
    max_seq_len: int = 128
    multi_vector: bool = False

    @torch.no_grad()
    def encode(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        self.model.eval()
        device = next(self.model.parameters()).device
        unique = list(dict.fromkeys(sentences))
        cache: Dict[str, torch.Tensor] = {}

        for start in tqdm(range(0, len(unique), self.batch_size), desc=f"{self.name} encode"):
            batch = unique[start : start + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_seq_len, return_tensors="pt",
            ).to(device)
            hidden = self.model(**enc).last_hidden_state
            pooled = F.normalize(pool(hidden, enc["attention_mask"], mode=self.pool_mode), dim=-1)
            for i, sent in enumerate(batch):
                cache[sent] = pooled[i : i + 1].cpu()

        return cache
