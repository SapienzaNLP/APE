"""
Configs for the fixed-size distillation baseline (w/o APE architecture).
"""

from dataclasses import dataclass, field

from ape.training.config import DataConfig


@dataclass
class BaselineTrainingConfig:
    """
    Config parameters for the baseline training loop.
    """
    pool_mode: str = "mean"
    train_batch_size: int = 64
    eval_batch_size: int = 256
    lr: float = 2e-5
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    num_epochs: int = 3
    max_seq_len: int = 128
    max_atom_seq_len: int = 64
    grad_clip: float = 1.0
    use_amp: bool = True

    # Logging and checkpointing
    log_every: int = 50
    eval_every: int = 2000

    # `output_dir` holds the fine-tuned encoder in `transformers` format
    # (loadable with `AutoModel.from_pretrained`) and `training_state.pt` for resuming.
    output_dir: str = "checkpoints/baseline-small"
    resume: bool = True


@dataclass
class BaselineConfig:
    """
    General config for baseline training (encoder + data + training loop).
    """
    encoder_name: str = "BAAI/bge-small-en-v1.5"
    data: DataConfig = field(default_factory=DataConfig)
    train: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    seed: int = 42
