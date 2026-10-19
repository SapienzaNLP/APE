"""
APE training entry point.

Usage:
    python -m ape.training.train
"""

import torch
from transformers import get_cosine_schedule_with_warmup

from ape.model import APE
from ape.training import loop
from ape.training.config import TrainConfig
from ape.training.data import build_loaders
from ape.training.loss import APELoss
from ape.training.teacher import load_frozen_teacher
from ape.training.utils import set_seed


def train(cfg: TrainConfig) -> APE:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Dataloaders
    train_loader, val_loader, test_loader = build_loaders(
        cfg.data, cfg.train, encoder_name=cfg.model.encoder_name, seed=cfg.seed
    )

    # Load (frozen) teacher and init APE, with the teacher's weights as its backbone
    teacher, _ = load_frozen_teacher(cfg.model.encoder_name, device)
    model = APE(cfg.model)
    model.encoder.load_state_dict(teacher.state_dict())
    model = model.to(device)
    model.freeze_backbone()

    # Criterion, optimizer and scheduler
    criterion = APELoss(
        emb_loss_weight=cfg.train.emb_loss_weight,
        act_loss_weight=cfg.train.act_loss_weight,
        rep_loss_weight=cfg.train.rep_loss_weight,
        activity_threshold=cfg.train.activity_threshold,
    )
    optimizer = loop.build_optimizer(
        model,
        lr_encoder=cfg.train.lr_encoder,
        lr_decoder=cfg.train.lr_decoder,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=len(train_loader) * cfg.train.num_epochs,
    )
    scaler = torch.amp.GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")

    # Checkpointing
    global_step = loop.maybe_load_ckpt(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        output_dir=cfg.train.output_dir,
        resume=cfg.train.resume,
    )

    # Training loop
    for epoch in range(cfg.train.num_epochs):
        global_step = loop.train_one_epoch(
            model,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            global_step=global_step,
            epoch=epoch,
            cfg=cfg.train,
        )

    # End-of-training evaluation
    for prefix, loader in (("val", val_loader), ("test", test_loader)):
        metrics = loop.evaluate(
            model,
            teacher=teacher,
            loader=loader,
            criterion=criterion,
            activity_threshold=cfg.train.activity_threshold,
            device=device,
            pool_mode=cfg.train.pool_mode,
            prefix=prefix,
        )
        loop.print_metrics(metrics, f"final eval | {prefix}", prefix=prefix)

    loop.save_ckpt(model, optimizer, scheduler, scaler, global_step, cfg.train.output_dir)
    print(f"[done] model saved to {cfg.train.output_dir}")
    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    print(cfg)
    train(cfg)
