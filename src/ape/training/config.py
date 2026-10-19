"""
Configs for APE training (data + optimization). The architecture is `ape.model.APEConfig`.
"""

from dataclasses import dataclass, field

from ape.model import APEConfig


@dataclass
class DataConfig:
    """
    Config parameters for the training dataset.
    """
    dataset_name: str = "francescoortame/atomic-wikisplit"  # HF repo id or local `save_to_disk` dir
    nli_filter: bool = True
    nli_filter_feature_name: str = "entailed"
    max_atoms: int = 5
    val_size: int = 5000
    test_size: int = 5000
    # Seeds the train/val/test split and the singleton sub-sampling, independently
    # of `TrainConfig.seed`, so the held-out sets stay fixed across training runs.
    data_seed: int = 42
    n_singletons: int | None = 20_000
    # If False, singletons (len(atoms) == 1) use the sentence itself as their target atom.
    use_paraphrased_singletons: bool = False


@dataclass
class TrainingConfig:
    """
    Config parameters for the training loop.
    """
    pool_mode: str = "mean"  # teacher pooling for target atom embeddings: "mean" or "cls"
    train_batch_size: int = 64
    eval_batch_size: int = 256
    lr_encoder: float = 2e-5
    lr_decoder: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    num_epochs: int = 3
    max_seq_len: int = 128
    max_atom_seq_len: int = 64
    encoder_unfreeze_step: int = 3000
    emb_loss_weight: float = 1.0
    act_loss_weight: float = 1.0
    rep_loss_weight: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True

    # Logging and checkpointing
    activity_threshold: float = 0.5  # only used for logging and the repulsion loss mask
    log_every: int = 50
    eval_every: int = 2000

    # `output_dir` holds the model for HF hub (config.json + model.safetensors)
    # and `training_state.pt` (optimizer, scheduler, step) for resuming.
    output_dir: str = "checkpoints/ape-model"
    resume: bool = True


@dataclass
class TrainConfig:
    """
    General config for APE training (model + data + training loop).
    """
    model: APEConfig = field(default_factory=APEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
