"""
PAWS benchmark: paraphrase detection (labeled_final subset).
Calibrates on validation split, evaluates on test split.
"""

from typing import Dict, Sequence

import numpy as np
from datasets import load_dataset

from ape.evaluation.utils import Pairs, run_pair_classification

HF_PATH = "paws"
SUBSET = "labeled_final"


def load_pairs(split: str) -> Pairs:
    """
    Labels: 1 = paraphrase, 0 = non-paraphrase.
    """
    ds = load_dataset(HF_PATH, SUBSET, split=split)
    return list(ds["sentence1"]), list(ds["sentence2"]), np.array(ds["label"], dtype=np.int32)


def run(encoders: Sequence) -> Dict[str, dict]:
    return run_pair_classification(
        "PAWS", load_pairs("validation"), load_pairs("test"), encoders, pos_label="paraphrase"
    )
