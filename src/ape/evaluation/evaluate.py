"""
Evaluation orchestrator: runs every benchmark selected in `EvalConfig`.

Usage:
    python -m ape.evaluation.evaluate
"""

from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

from ape.evaluation import count_analysis
from ape.evaluation.benchmarks import paws, sickr, snli, stsb
from ape.evaluation.config import EvalConfig
from ape.evaluation.encoders import APEEncoder, PooledEncoder
from ape.model import APE
from ape.propsearch import benchmark as propsearch

PAIR_BENCHMARKS = {"paws": paws.run, "snli": snli.run, "stsb": stsb.run, "sickr": sickr.run}


def _load_pooled(name: str, path: str, pool_mode: str, cfg: EvalConfig, device: torch.device) -> PooledEncoder:
    print(f"[load] {name}: {path} ({pool_mode} pooling)")
    return PooledEncoder(
        name=name,
        model=AutoModel.from_pretrained(path).to(device).eval(),
        tokenizer=AutoTokenizer.from_pretrained(path),
        pool_mode=pool_mode,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
    )


def load_encoders(cfg: EvalConfig, device: torch.device) -> List:
    """
    APE first, then the fixed-size baselines selected in `cfg`.
    """
    print(f"[load] APE: {cfg.ape_model}")
    model = APE.from_pretrained(cfg.ape_model).to(device).eval()
    encoders = [
        APEEncoder(
            model,
            activity_threshold=cfg.activity_threshold,
            batch_size=cfg.batch_size,
            max_seq_len=cfg.max_seq_len,
        )
    ]

    if cfg.include_teacher:
        teacher = _load_pooled("teacher_cls", model.config.encoder_name, "cls", cfg, device)
        encoders.append(teacher)
        encoders.append(
            PooledEncoder("teacher_mean", teacher.model, teacher.tokenizer, "mean", cfg.batch_size, cfg.max_seq_len)
        )
    if cfg.baseline_model is not None:
        encoders.append(_load_pooled("baseline", cfg.baseline_model, cfg.baseline_pool_mode, cfg, device))
    return encoders


def evaluate(cfg: EvalConfig) -> Dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    unknown = set(cfg.benchmarks) - set(PAIR_BENCHMARKS) - {"propsearch", "count_analysis"}
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")

    encoders = load_encoders(cfg, device)
    results: Dict[str, dict] = {}
    for name in cfg.benchmarks:
        print(f"\n{'#' * 80}\n# {name}\n{'#' * 80}")
        if name in PAIR_BENCHMARKS:
            results[name] = PAIR_BENCHMARKS[name](encoders)
        elif name == "propsearch":
            results[name] = propsearch.run(
                encoders,
                device,
                repo_id=cfg.propsearch_repo_id,
                all_rules=cfg.propsearch_all_rules,
                chunk_size=cfg.propsearch_chunk_size,
            )
        elif name == "count_analysis":
            results[name] = count_analysis.run(
                encoders[0].model,
                cfg.count_analysis,
                propsearch_repo_id=cfg.propsearch_repo_id,
                batch_size=cfg.batch_size,
                max_seq_len=cfg.max_seq_len,
            )
    return results


if __name__ == "__main__":
    cfg = EvalConfig()
    print(cfg)
    evaluate(cfg)
