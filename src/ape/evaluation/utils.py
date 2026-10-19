"""
Shared benchmark utilities: pair scoring, threshold calibration and reporting.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from ape.scoring import APEScore

Pairs = Tuple[List[str], List[str], np.ndarray]  # (sents_a, sents_b, labels or gold scores)


def scorer_names(encoders: Sequence) -> List[str]:
    """
    One row per APE scoring rule for multi-vector encoders, one row (cosine) for pooled ones.
    """
    names = []
    for enc in encoders:
        if enc.multi_vector:
            names += [f"{enc.name}_{rule}" for rule in APEScore.RULES]
        else:
            names.append(enc.name)
    return names


def encode_all(encoders: Sequence, sentences: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {enc.name: enc.encode(sentences) for enc in encoders}


def score_pairs(
    sents_a: List[str],
    sents_b: List[str],
    encoders: Sequence,
    caches: Dict[str, Dict[str, torch.Tensor]],
) -> dict:
    """
    Score all pairs with every encoder (every APE rule for multi-vector encoders).

    For NLI benchmarks, pass sents_a=hypotheses and sents_b=premises so that
    the asymmetric APEScore(emb_a, emb_b) = APEScore(emb_h, emb_p) convention holds.

    Returns:
        scores:   dict[name, float array of length n_pairs]
        counts_a: proposition counts of sents_a (first multi-vector encoder)
        counts_b: proposition counts of sents_b (first multi-vector encoder)
    """
    raw: Dict[str, list] = {n: [] for n in scorer_names(encoders)}
    count_cache = next(caches[e.name] for e in encoders if e.multi_vector)
    counts_a, counts_b = [], []

    for a, b in tqdm(zip(sents_a, sents_b), total=len(sents_a), desc="scoring"):
        for enc in encoders:
            ea, eb = caches[enc.name][a], caches[enc.name][b]
            if enc.multi_vector:
                scorer = APEScore(ea, eb)
                for rule in APEScore.RULES:
                    raw[f"{enc.name}_{rule}"].append(scorer.score(rule))
            else:
                raw[enc.name].append(float((ea @ eb.T).item()))
        counts_a.append(count_cache[a].size(0))
        counts_b.append(count_cache[b].size(0))

    return {
        "scores": {n: np.array(v, dtype=np.float64) for n, v in raw.items()},
        "counts_a": np.array(counts_a, dtype=np.int32),
        "counts_b": np.array(counts_b, dtype=np.int32),
    }


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Sweep score thresholds, return (best_thr, best_acc, best_f1).
    """
    candidates = np.unique(scores)
    if len(candidates) > 500:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 500).astype(int)]
    best_f1, best_thr, best_acc = -1.0, 0.0, 0.0
    for thr in candidates:
        preds = (scores >= thr).astype(np.int32)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_acc = accuracy_score(labels, preds)
    return best_thr, best_acc, best_f1


def report_classification(
    val_res: dict,
    test_res: dict,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    names: List[str],
    bench_name: str,
    pos_label: str = "positive",
) -> Dict[str, dict]:
    """
    Threshold-tune on val (F1), report AUC/AP/acc/F1 on test, overall and on the
    multi-proposition slice (either sentence has >= 2 APE propositions).
    """
    multi_mask = (test_res["counts_a"] >= 2) | (test_res["counts_b"] >= 2)
    all_mask = np.ones(len(test_labels), dtype=bool)

    n = len(test_labels)
    print(
        f"\n--- {bench_name} test (n={n}, {test_labels.mean() * 100:.1f}% {pos_label} "
        f"| multi-prop n={int(multi_mask.sum())}) ---"
    )
    print(f"{'scorer':<28} {'AUC':>6} {'AP':>6} {'acc':>6} {'F1':>6} | {'mAUC':>6} {'mAP':>6} {'mAcc':>6} {'mF1':>6}")
    print("-" * 86)

    def _metrics(sc, lb, preds, mask):
        if mask.sum() < 10 or len(np.unique(lb[mask])) < 2:
            return None
        return {
            "auc": roc_auc_score(lb[mask], sc[mask]),
            "ap":  average_precision_score(lb[mask], sc[mask]),
            "acc": accuracy_score(lb[mask], preds[mask]),
            "f1":  f1_score(lb[mask], preds[mask], zero_division=0),
        }

    def _fmt(m, key):
        return "    --" if m is None else f"{m[key] * 100:>6.2f}"

    results = {}
    for name in names:
        test_sc = test_res["scores"][name]
        thr, _, _ = calibrate_threshold(val_res["scores"][name], val_labels)
        preds = (test_sc >= thr).astype(np.int32)
        overall = _metrics(test_sc, test_labels, preds, all_mask)
        multi = _metrics(test_sc, test_labels, preds, multi_mask)
        results[name] = {"overall": overall, "multi": multi, "threshold": thr}
        print(
            f"{name:<28} "
            f"{_fmt(overall, 'auc')} {_fmt(overall, 'ap')} {_fmt(overall, 'acc')} {_fmt(overall, 'f1')} | "
            f"{_fmt(multi, 'auc')} {_fmt(multi, 'ap')} {_fmt(multi, 'acc')} {_fmt(multi, 'f1')}"
        )
    return results


def _corr(preds: np.ndarray, gold: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Returns (spearman, pearson) for the masked subset, or (nan, nan) if too small.
    """
    if mask.sum() < 2:
        return float("nan"), float("nan")
    return (
        float(spearmanr(preds[mask], gold[mask]).statistic),
        float(pearsonr(preds[mask], gold[mask]).statistic),
    )


def report_correlation(
    res: dict,
    gold: np.ndarray,
    names: List[str],
    bench_name: str,
) -> Dict[str, dict]:
    """
    Spearman/Pearson report for graded-similarity benchmarks (overall / multi-prop / single-prop).
    """
    multi_mask = (res["counts_a"] >= 2) | (res["counts_b"] >= 2)
    single_mask = ~multi_mask
    all_mask = np.ones(len(gold), dtype=bool)

    n_total, n_multi = len(gold), int(multi_mask.sum())
    print(f"\n--- {bench_name} test (n={n_total} | multi-prop n={n_multi} | single-prop n={n_total - n_multi}) ---")
    print(f"{'scorer':<28} {'all sp':>7} {'all pe':>7} | {'multi sp':>9} {'multi pe':>9} | {'single sp':>10} {'single pe':>10}")
    print("-" * 90)

    def _fmt(v: float, w: int) -> str:
        return f"{'--':>{w}}" if np.isnan(v) else f"{v * 100:>{w}.2f}"

    results = {}
    for name in names:
        preds = res["scores"][name]
        sp_a, pe_a = _corr(preds, gold, all_mask)
        sp_m, pe_m = _corr(preds, gold, multi_mask)
        sp_s, pe_s = _corr(preds, gold, single_mask)
        results[name] = {
            "overall": {"spearman": sp_a, "pearson": pe_a},
            "multi": {"spearman": sp_m, "pearson": pe_m},
            "single": {"spearman": sp_s, "pearson": pe_s},
        }
        print(
            f"{name:<28} "
            f"{_fmt(sp_a, 7)} {_fmt(pe_a, 7)} | "
            f"{_fmt(sp_m, 9)} {_fmt(pe_m, 9)} | "
            f"{_fmt(sp_s, 10)} {_fmt(pe_s, 10)}"
        )
    return results


def run_pair_classification(
    bench_name: str,
    val: Pairs,
    test: Pairs,
    encoders: Sequence,
    pos_label: str,
) -> Dict[str, dict]:
    """
    Binary pair classification: calibrate each scorer's threshold on `val`, evaluate on `test`.
    """
    val_a, val_b, val_y = val
    test_a, test_b, test_y = test
    print(
        f"[{bench_name}] val={len(val_y)} ({val_y.mean() * 100:.1f}% pos)  "
        f"test={len(test_y)} ({test_y.mean() * 100:.1f}% pos)"
    )
    all_sents = list(dict.fromkeys(val_a + val_b + test_a + test_b))
    print(f"[{bench_name}] {len(all_sents)} unique sentences")

    caches = encode_all(encoders, all_sents)
    print(f"[{bench_name}] scoring ...")
    val_res = score_pairs(val_a, val_b, encoders, caches)
    test_res = score_pairs(test_a, test_b, encoders, caches)
    return report_classification(
        val_res, test_res, val_y, test_y, scorer_names(encoders), bench_name, pos_label
    )


def run_correlation(bench_name: str, pairs: Pairs, encoders: Sequence) -> Dict[str, dict]:
    """
    Graded similarity: Spearman/Pearson correlation of each scorer with the gold scores.
    """
    sents_a, sents_b, gold = pairs
    print(f"[{bench_name}] {len(gold)} pairs, score range [{gold.min():.1f}, {gold.max():.1f}]")
    all_sents = list(dict.fromkeys(sents_a + sents_b))
    print(f"[{bench_name}] {len(all_sents)} unique sentences")

    caches = encode_all(encoders, all_sents)
    print(f"[{bench_name}] scoring ...")
    res = score_pairs(sents_a, sents_b, encoders, caches)
    return report_correlation(res, gold, scorer_names(encoders), bench_name)
