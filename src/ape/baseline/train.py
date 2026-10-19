"""
Fixed-size distillation baseline (fine-tune the encoder to match the average of proposition embeddings).

For each (sentence, atoms) pair, the student is trained to produce a pooled
sentence embedding that is close (cosine) to the mean of the frozen teacher's
atom embeddings. Trained on the same data as APE, this aims to isolate the
effect of the slot architecture.

Usage:
    python -m ape.baseline.train
"""

import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, get_cosine_schedule_with_warmup

from ape.baseline.config import BaselineConfig
from ape.training.data import build_loaders
from ape.training.teacher import encode_gt_atoms, load_frozen_teacher, pool
from ape.training.utils import set_seed

TRAINING_STATE_NAME = "training_state.pt"


def _mean_of_atoms(
    tgt_embeds: torch.Tensor,
    atom_offsets: torch.Tensor,
    B: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (targets [B, D], valid_mask [B]) where valid_mask is False for
    sentences that have no atoms (shouldn't happen after data filtering).
    """
    D = tgt_embeds.size(-1)
    device = tgt_embeds.device
    targets = torch.zeros(B, D, device=device, dtype=tgt_embeds.dtype)
    valid = torch.zeros(B, dtype=torch.bool, device=device)
    for i in range(B):
        s, e = int(atom_offsets[i]), int(atom_offsets[i + 1])
        if e > s:
            targets[i] = tgt_embeds[s:e].mean(0)
            valid[i] = True
    return targets, valid


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def _maybe_load_ckpt(model, optimizer, scheduler, scaler, output_dir, resume) -> int:
    path = os.path.join(output_dir, TRAINING_STATE_NAME)
    if not (resume and os.path.isfile(path)):
        return 0
    print(f"[resume] loading checkpoint {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    if ckpt.get("sched") is not None:
        scheduler.load_state_dict(ckpt["sched"])
    if ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)


def _save_ckpt(model, tokenizer, optimizer, scheduler, scaler, step, output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
        },
        os.path.join(output_dir, TRAINING_STATE_NAME),
    )


@torch.no_grad()
def evaluate(model, teacher, loader, device, pool_mode) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        sid = batch["sentence_input_ids"].to(device)
        sam = batch["sentence_attention_mask"].to(device)
        aid = batch["atom_input_ids"].to(device)
        aam = batch["atom_attention_mask"].to(device)
        offsets = batch["atom_offsets"].to(device)

        out = model(input_ids=sid, attention_mask=sam)
        pred = pool(out.last_hidden_state, sam, mode=pool_mode)
        tgt_embeds = encode_gt_atoms(teacher, aid, aam, pool_mode)
        targets, valid = _mean_of_atoms(tgt_embeds, offsets, sid.size(0))

        if valid.any():
            total_loss += _cosine_loss(pred[valid], targets[valid]).item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def train(cfg: BaselineConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_loader, val_loader, _ = build_loaders(
        cfg.data, cfg.train, encoder_name=cfg.encoder_name, seed=cfg.seed
    )

    model = AutoModel.from_pretrained(cfg.encoder_name).to(device)
    teacher, tokenizer = load_frozen_teacher(cfg.encoder_name, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=len(train_loader) * cfg.train.num_epochs,
    )
    scaler = torch.amp.GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")

    global_step = _maybe_load_ckpt(
        model, optimizer, scheduler, scaler, cfg.train.output_dir, cfg.train.resume,
    )

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    n_epochs = cfg.train.num_epochs
    for epoch in range(n_epochs):
        model.train()
        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            sid = batch["sentence_input_ids"].to(device, non_blocking=True)
            sam = batch["sentence_attention_mask"].to(device, non_blocking=True)
            aid = batch["atom_input_ids"].to(device, non_blocking=True)
            aam = batch["atom_attention_mask"].to(device, non_blocking=True)
            offsets = batch["atom_offsets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.train.use_amp):
                out = model(input_ids=sid, attention_mask=sam)
                pred = pool(out.last_hidden_state, sam, mode=cfg.train.pool_mode)
                tgt_embeds = encode_gt_atoms(teacher, aid, aam, cfg.train.pool_mode)
                targets, valid = _mean_of_atoms(tgt_embeds, offsets, sid.size(0))
                loss = _cosine_loss(pred[valid], targets[valid]) if valid.any() else pred.sum() * 0.0

            if cfg.train.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % cfg.train.log_every == 0:
                print(
                    f"[epoch {epoch + 1}/{n_epochs} | step {global_step} | batch {batch_idx + 1}/{n_batches}] "
                    f"loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step % cfg.train.eval_every == 0:
                val_loss = evaluate(model, teacher, val_loader, device, cfg.train.pool_mode)
                print(f"\n[eval @ step {global_step}] val_loss={val_loss:.4f}\n")
                _save_ckpt(model, tokenizer, optimizer, scheduler, scaler, global_step, cfg.train.output_dir)
                model.train()

    val_loss = evaluate(model, teacher, val_loader, device, cfg.train.pool_mode)
    print(f"\n[final eval] val_loss={val_loss:.4f}")
    _save_ckpt(model, tokenizer, optimizer, scheduler, scaler, global_step, cfg.train.output_dir)
    print(f"[done] model saved to {cfg.train.output_dir}")


if __name__ == "__main__":
    cfg = BaselineConfig()
    print(cfg)
    train(cfg)
