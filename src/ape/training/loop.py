"""
Training loop logic - eval, optimizer, checkpointing, epoch.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ape.model import APE
from ape.training.config import TrainingConfig
from ape.training.loss import APELoss
from ape.training.teacher import encode_gt_atoms
from ape.training.utils import format_confusion

TRAINING_STATE_NAME = "training_state.pt"


@torch.no_grad()
def evaluate(
    model: APE,
    teacher: nn.Module,
    loader: DataLoader,
    criterion: APELoss,
    activity_threshold: float,
    device: torch.device,
    pool_mode: str = "mean",
    prefix: str = "val",
) -> dict:
    model.eval()
    teacher.eval()
    num_slots = model.config.num_slots

    total_emb_loss: float = 0.0
    total_act_loss: float = 0.0
    total_rep_loss: float = 0.0
    total_matched: int = 0
    n_batches: int = 0

    # Initialize activity confusion matrix for logging
    confusion = np.zeros((num_slots + 1, num_slots + 1), dtype=np.int64)

    aa_cos_sum: float = 0.0
    aa_cos_count: int = 0
    ai_cos_sum: float = 0.0
    ai_cos_count: int = 0

    for batch in loader:
        sid = batch["sentence_input_ids"].to(device)
        sam = batch["sentence_attention_mask"].to(device)
        aid = batch["atom_input_ids"].to(device)
        aam = batch["atom_attention_mask"].to(device)
        offsets = batch["atom_offsets"].to(device)
        num_atoms = batch["num_atoms"].to(device)

        slot_embeds, slot_act_logits = model(sid, sam)
        tgt_embeds = encode_gt_atoms(teacher, aid, aam, pool_mode)

        _, _, stats = criterion(slot_embeds, slot_act_logits, tgt_embeds, offsets)
        total_emb_loss += stats["loss_emb"]
        total_act_loss += stats["loss_act"]
        total_rep_loss += stats["loss_rep"]
        total_matched += stats["matched"]
        n_batches += 1

        pred_counts = (torch.sigmoid(slot_act_logits) > activity_threshold).sum(dim=-1).cpu().numpy()
        true_counts = num_atoms.cpu().numpy()
        for t, p in zip(true_counts, pred_counts):
            t_c = min(int(t), num_slots)
            p_c = min(int(p), num_slots)
            confusion[t_c, p_c] += 1

        slots_norm = F.normalize(slot_embeds, dim=-1)
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))

        act_probs = torch.sigmoid(slot_act_logits)
        active = act_probs > activity_threshold
        off_diag = ~torch.eye(num_slots, dtype=torch.bool, device=device)

        # active-active pairs (off-diagonal)
        aa_mask = active.unsqueeze(2) & active.unsqueeze(1) & off_diag.unsqueeze(0)
        if aa_mask.any():
            aa_cos_sum += sim[aa_mask].mean().item() * sid.size(0)
            aa_cos_count += sid.size(0)

        # active-inactive pairs (both directions: active->inactive and inactive->active)
        ai_mask = (active.unsqueeze(2) ^ active.unsqueeze(1)) & off_diag.unsqueeze(0)
        if ai_mask.any():
            ai_cos_sum += sim[ai_mask].mean().item() * sid.size(0)
            ai_cos_count += sid.size(0)

    return {
        f"{prefix}_loss_emb": total_emb_loss / max(1, n_batches),
        f"{prefix}_loss_act": total_act_loss / max(1, n_batches),
        f"{prefix}_loss_rep": total_rep_loss / max(1, n_batches),
        f"{prefix}_matched": total_matched,
        f"{prefix}_slot_cos_active_active":   aa_cos_sum / max(1, aa_cos_count),
        f"{prefix}_slot_cos_active_inactive": ai_cos_sum / max(1, ai_cos_count),
        f"{prefix}_cardinality_confusion": confusion,
    }


def print_metrics(metrics: dict, header: str, prefix: str = "val") -> None:
    print(
        f"\n[{header}] emb={metrics[f'{prefix}_loss_emb']:.4f} "
        f"act={metrics[f'{prefix}_loss_act']:.4f} "
        f"rep={metrics[f'{prefix}_loss_rep']:.4f} "
        f"cos_aa={metrics[f'{prefix}_slot_cos_active_active']:.4f} "
        f"cos_ai={metrics[f'{prefix}_slot_cos_active_inactive']:.4f}"
    )
    print(f"[cardinality confusion | {prefix}]")
    print(format_confusion(metrics[f"{prefix}_cardinality_confusion"]))


def build_optimizer(
    model: APE,
    lr_encoder: float,
    lr_decoder: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": decoder_params, "lr": lr_decoder},
        ],
        weight_decay=weight_decay,
    )


def maybe_load_ckpt(
    model: APE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    output_dir: str,
    resume: bool,
) -> int:
    """
    Loads the training state from `output_dir` if it exists. Returns starting global step.
    """
    path = os.path.join(output_dir, TRAINING_STATE_NAME)
    if not (resume and os.path.isfile(path)):
        return 0
    print(f"[resume] loading checkpoint {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    if scheduler is not None and ckpt.get("sched") is not None:
        scheduler.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("encoder_unfrozen", False):
        model.unfreeze_backbone()
    return ckpt.get("step", 0)


def save_ckpt(
    model: APE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    output_dir: str,
) -> None:
    """
    Saves the Hub-ready model (`save_pretrained`) and the full training state to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "encoder_unfrozen": not model.encoder_frozen,
        },
        os.path.join(output_dir, TRAINING_STATE_NAME),
    )


def train_one_epoch(
    model: APE,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: APELoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    global_step: int,
    epoch: int,
    cfg: TrainingConfig,
) -> int:
    """
    Runs one epoch of training. Returns updated global_step.
    """
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    model.train()

    n_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        if model.encoder_frozen and global_step >= cfg.encoder_unfreeze_step:
            print(f"[unfreeze] unfreezing encoder at step {global_step}")
            model.unfreeze_backbone()

        sid = batch["sentence_input_ids"].to(device, non_blocking=True)
        sam = batch["sentence_attention_mask"].to(device, non_blocking=True)
        aid = batch["atom_input_ids"].to(device, non_blocking=True)
        aam = batch["atom_attention_mask"].to(device, non_blocking=True)
        offsets = batch["atom_offsets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.use_amp):
            slot_embeds, slot_act_logits = model(sid, sam)
            tgt_embeds = encode_gt_atoms(teacher, aid, aam, cfg.pool_mode)
            loss, _, stats = criterion(slot_embeds, slot_act_logits, tgt_embeds, offsets)

        if cfg.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        scheduler.step()
        global_step += 1

        if global_step % cfg.log_every == 0:
            print(
                f"[epoch {epoch + 1}/{cfg.num_epochs} | step {global_step} | batch {batch_idx + 1}/{n_batches}] "
                f"loss={stats['loss_total']:.4f} "
                f"emb={stats['loss_emb']:.4f} "
                f"act={stats['loss_act']:.4f} "
                f"rep={stats['loss_rep']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if global_step % cfg.eval_every == 0:
            metrics = evaluate(
                model, teacher, val_loader, criterion,
                activity_threshold=cfg.activity_threshold,
                device=device,
                pool_mode=cfg.pool_mode,
            )
            print_metrics(metrics, f"eval @ step {global_step}")
            save_ckpt(model, optimizer, scheduler, scaler, global_step, cfg.output_dir)
            model.train()

    return global_step
