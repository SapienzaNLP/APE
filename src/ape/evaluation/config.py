"""
Configs for APE evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from ape.propsearch.data import REPO_ID as PROPSEARCH_REPO_ID
from ape.training.config import DataConfig


@dataclass
class CountAnalysisConfig:
    """
    Proposition-count analysis, in-domain (held-out test split of the training data,
    see `ape.training.data.load_splits`) and out-of-domain (PropSearch "counting" subset).
    """
    data: DataConfig = field(default_factory=DataConfig)
    # The in-domain test split is further divided into a calibration part (threshold
    # sweep) and an evaluation part (fixed threshold), to avoid selection leakage.
    calibration_fraction: float = 0.4
    threshold_grid: Tuple[float, float, int] = (0.05, 0.95, 37)  # np.linspace args


@dataclass
class EvalConfig:
    """
    Config for `ape.evaluation.evaluate`.
    """
    # APE checkpoint: Hub repo id or local directory (see `APE.from_pretrained`)
    ape_model: str = "francescoortame/ape-bge-small"
    activity_threshold: Optional[float] = None  # None -> default (see config in model.py)

    # Fixed-size baselines
    include_teacher: bool = True               # APE's teacher (both cls and mean pooling)
    baseline_model: Optional[str] = None       # fixed-size baseline (output of `ape.baseline.train`)
    baseline_pool_mode: str = "mean"

    # Any of: "paws", "snli", "stsb", "sickr", "propsearch", "count_analysis"
    benchmarks: Tuple[str, ...] = ("paws", "snli", "stsb", "sickr", "propsearch", "count_analysis")
    batch_size: int = 128
    max_seq_len: int = 128

    # PropSearch
    propsearch_repo_id: str = PROPSEARCH_REPO_ID
    propsearch_all_rules: bool = False  # also score APE with maxsim and pool (slower)
    propsearch_chunk_size: int = 128

    count_analysis: CountAnalysisConfig = field(default_factory=CountAnalysisConfig)
