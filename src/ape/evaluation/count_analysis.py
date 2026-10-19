"""
Proposition-count analysis.

How well does APE's number of active slots matche the gold atom count?

Evaluated in-domain, on the held-out test split of the training data, and
out-of-domain, on the PropSearch "counting" subset. The activity threshold is
swept and chosen as the mean of the best-exact-match and best-MAE thresholds
(e.g. 0.55 and 0.65 -> 0.6). In-domain results are reported both swept and
evaluated on the full test split, and swept on a calibration part / evaluated on
the remaining part (leakage-free); the leakage-free threshold is then applied,
fixed, out-of-domain.

Usage:
    python -m ape.evaluation.count_analysis
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ape.evaluation.config import CountAnalysisConfig, EvalConfig
from ape.model import APE
from ape.propsearch.data import REPO_ID, load_counting
from ape.training.data import load_splits


def _sweep(probs: np.ndarray, true_counts: np.ndarray, thresholds: np.ndarray) -> dict:
    """
    Sweep activity thresholds and compute count-accuracy metrics.
    """
    res = {k: np.empty(len(thresholds)) for k in
           ("mean_active", "exact_match", "pm1_match", "mae", "over_rate", "under_rate")}
    res["thresholds"] = thresholds
    for i, thr in enumerate(thresholds):
        pred = (probs > thr).sum(axis=1).astype(int)
        diff = pred - true_counts.astype(int)
        res["mean_active"][i] = pred.mean()
        res["exact_match"][i] = (diff == 0).mean()
        res["pm1_match"][i] = (np.abs(diff) <= 1).mean()
        res["mae"][i] = np.abs(diff).mean()
        res["over_rate"][i] = (diff > 0).mean()
        res["under_rate"][i] = (diff < 0).mean()
    return res


def _report_sweep(res: dict) -> None:
    """
    Print the full threshold sweep table with best-threshold markers.
    """
    thr, exact, mae = res["thresholds"], res["exact_match"], res["mae"]
    best_exact_idx, best_mae_idx = int(np.argmax(exact)), int(np.argmin(mae))
    markers = {best_exact_idx: " <- best exact", best_mae_idx: " <- best MAE"}
    if best_exact_idx == best_mae_idx:
        markers[best_exact_idx] = " <- best exact & MAE"

    print(f"\n  {'thr':>5}  {'mean_act':>8}  {'exact%':>7}  {'+-1%':>7}  {'MAE':>6}  {'over%':>6}  {'under%':>7}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*7}")
    for i, thr_i in enumerate(thr):
        print(
            f"  {thr_i:>5.2f}  {res['mean_active'][i]:>8.3f}  "
            f"{100*exact[i]:>7.2f}  {100*res['pm1_match'][i]:>7.2f}  "
            f"{mae[i]:>6.3f}  "
            f"{100*res['over_rate'][i]:>6.2f}  {100*res['under_rate'][i]:>7.2f}"
            f"{markers.get(i, '')}"
        )


def _report_per_natom(pred: np.ndarray, true_counts: np.ndarray, threshold: float) -> dict:
    """
    Summary and per-n_atom accuracy at a fixed threshold.
    """
    true = true_counts.astype(int)
    diff = pred - true
    summary = {
        "threshold": threshold,
        "mean_pred": float(pred.mean()),
        "exact": float((diff == 0).mean()),
        "pm1": float((np.abs(diff) <= 1).mean()),
        "mae": float(np.abs(diff).mean()),
        "over": float((diff > 0).mean()),
        "under": float((diff < 0).mean()),
    }
    print(
        f"\n-- thr={threshold:.3f}  mean_pred={summary['mean_pred']:.3f}  "
        f"exact={100*summary['exact']:.2f}%  +-1={100*summary['pm1']:.2f}%  MAE={summary['mae']:.3f}  "
        f"over={100*summary['over']:.2f}%  under={100*summary['under']:.2f}% --"
    )
    print(f"  {'n_atom':>6}  {'N':>5}  {'%total':>6}  {'exact%':>7}  {'+-1%':>7}")
    sep = f"  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}"
    print(sep)
    for n in np.sort(np.unique(true)):
        mask = true == n
        d = diff[mask]
        print(
            f"  {n:>6}  {int(mask.sum()):>5}  {100*mask.mean():>6.1f}  "
            f"{100*(d == 0).mean():>7.2f}  {100*(np.abs(d) <= 1).mean():>7.2f}"
        )
    print(sep)
    print(f"  {'total':>6}  {len(true):>5}  {'100.0':>6}  {100*summary['exact']:>7.2f}  {100*summary['pm1']:>7.2f}")
    return summary


def analyse(
    label: str,
    probs: np.ndarray,
    true_counts: np.ndarray,
    thresholds: np.ndarray,
    fixed_threshold: Optional[float] = None,
) -> dict:
    """
    If `fixed_threshold` is None, sweep `thresholds` and select mean(best exact, best MAE);
    otherwise evaluate at `fixed_threshold` only.
    """
    print(f"\n{'='*70}")
    print(f"  APE  {label}  |  N={len(true_counts)}  true count range [{true_counts.min()}, {true_counts.max()}]")
    print(f"{'='*70}")

    if fixed_threshold is None:
        res = _sweep(probs, true_counts, thresholds)
        _report_sweep(res)
        best_exact = thresholds[int(np.argmax(res["exact_match"]))]
        best_mae = thresholds[int(np.argmin(res["mae"]))]
        threshold = float(0.5 * (best_exact + best_mae))
        print(f"\n  -> selected threshold = mean(best_exact={best_exact:.3f}, best_mae={best_mae:.3f}) = {threshold:.3f}")
    else:
        threshold = fixed_threshold

    pred = (probs > threshold).sum(axis=1).astype(int)
    return _report_per_natom(pred, true_counts, threshold)


def run(
    model: APE,
    cfg: Optional[CountAnalysisConfig] = None,
    propsearch_repo_id: str = REPO_ID,
    batch_size: int = 128,
    max_seq_len: int = 128,
) -> dict:
    cfg = cfg or CountAnalysisConfig()
    thresholds = np.linspace(*cfg.threshold_grid)

    def probs_of(sentences: List[str]) -> np.ndarray:
        return model.activity_scores(
            sentences, batch_size=batch_size, max_seq_len=max_seq_len, show_progress_bar=True
        )

    results = {}

    # In-domain: held-out test split of the training data (already filtered to 1...max_atoms)
    _, _, test = load_splits(cfg.data)
    sentences = list(test["sentence"])
    counts = np.array([len(a) for a in test["atoms"]])
    probs = probs_of(sentences)
    results["in_domain_full"] = analyse("IN-DOMAIN  (test, full — sweep & eval on same set)", probs, counts, thresholds)

    rng = np.random.default_rng(cfg.data.data_seed)
    perm = rng.permutation(len(sentences))
    n_cal = max(1, int(len(sentences) * cfg.calibration_fraction))
    cal, ev = perm[:n_cal], perm[n_cal:]
    print(f"\n[in-domain split] cal={len(cal)}  eval={len(ev)}")
    cal_res = analyse("IN-DOMAIN  (test, calibration split)", probs[cal], counts[cal], thresholds)
    results["in_domain_calibration"] = cal_res
    results["in_domain_eval"] = analyse(
        "IN-DOMAIN  (test, evaluation split)", probs[ev], counts[ev], thresholds,
        fixed_threshold=cal_res["threshold"],
    )

    # Out-of-domain: PropSearch counting subset, same atom-count range as training
    ood_sents, ood_atoms = load_counting(propsearch_repo_id)
    keep = [i for i, a in enumerate(ood_atoms) if 1 <= len(a) <= cfg.data.max_atoms]
    print(f"\n[ood] raw={len(ood_sents)}  filtered={len(keep)}")
    ood_probs = probs_of([ood_sents[i] for i in keep])
    ood_counts = np.array([len(ood_atoms[i]) for i in keep])
    results["out_of_domain"] = analyse(
        "OUT-OF-DOMAIN  (PropSearch)", ood_probs, ood_counts, thresholds,
        fixed_threshold=cal_res["threshold"],
    )
    return results


if __name__ == "__main__":
    cfg = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APE.from_pretrained(cfg.ape_model).to(device)
    run(model, cfg.count_analysis, cfg.propsearch_repo_id, cfg.batch_size, cfg.max_seq_len)
