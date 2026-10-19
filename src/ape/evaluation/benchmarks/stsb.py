"""
STS-B benchmark: graded semantic similarity (scores 0-5).
Single test split; reports Spearman and Pearson correlation.
"""

from typing import Dict, Sequence

import numpy as np
from datasets import load_dataset

from ape.evaluation.utils import Pairs, run_correlation

HF_PATH = "mteb/stsbenchmark-sts"


def load_pairs(split: str = "test") -> Pairs:
    ds = load_dataset(HF_PATH, split=split)
    return list(ds["sentence1"]), list(ds["sentence2"]), np.array(ds["score"], dtype=np.float64)


def run(encoders: Sequence) -> Dict[str, dict]:
    return run_correlation("STS-B", load_pairs(), encoders)
