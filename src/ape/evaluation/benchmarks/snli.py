"""
SNLI benchmark: binary entailment (entailment vs. neutral+contradiction).
Calibrates on validation split, evaluates on test split.
"""

from typing import Dict, Sequence

import numpy as np
from datasets import load_dataset

from ape.evaluation.utils import Pairs, run_pair_classification

HF_PATH = "stanfordnlp/snli"
LABEL_ENTAILMENT = 0


def load_pairs(split: str) -> Pairs:
    """
    Returns (hypotheses, premises, labels). Labels: 1 = entailment, 0 = rest.
    Pairs with label=-1 (no gold label) are filtered out.
    """
    ds = load_dataset(HF_PATH, split=split)
    ds = ds.filter(lambda ex: ex["label"] != -1)
    labels = (np.array(ds["label"], dtype=np.int32) == LABEL_ENTAILMENT).astype(np.int32)
    return list(ds["hypothesis"]), list(ds["premise"]), labels


def run(encoders: Sequence) -> Dict[str, dict]:
    # sents_a=hypotheses, sents_b=premises -> APEScore(emb_h, emb_p) orientation
    return run_pair_classification(
        "SNLI", load_pairs("validation"), load_pairs("test"), encoders, pos_label="entailment"
    )
