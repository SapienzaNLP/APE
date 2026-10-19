"""
PropSearch: atom-to-sentence retrieval benchmark.

For each query atom, retrieve its source sentence from a corpus padded with
distractor sentences. Reports Recall@1/5/10 and MRR, overall and grouped by
the source sentence's (original) atom count.

APE and SSE are scored with h_min_to_p (every query proposition must be covered
by some corpus proposition). Pooled encoders are scored with cosine similarity.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ape.propsearch.data import REPO_ID, load_propsearch

TAG = "[PropSearch]"


def _rank(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """
    Rank of the gold column in each row: 1 + #(scores strictly greater than the gold score).
    """
    gold_scores = scores.gather(1, gold.unsqueeze(1))
    return 1 + (scores > gold_scores).sum(dim=1)


def _multi_vector_ranks(
    q_cache: Dict[str, torch.Tensor],
    queries: List[str],
    c_cache: Dict[str, torch.Tensor],
    corpus: List[str],
    gold: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    all_rules: bool,
) -> Dict[str, np.ndarray]:
    """
    Gold ranks under the APE rules, without materializing the [N_q, N_c] score matrix.

    Rules:
      h_min_to_p — per q-slot max to any c-slot, then min over q-slots
      maxsim     — symmetric: 0.5*(fwd+bwd) mean-of-max          (only if all_rules)
      pool       — mean-pool both sides, cosine sim              (only if all_rules)

    Each chunk of queries is scored with one matmul against all corpus slots,
    followed by grouped scatter reductions.
    """
    c_slots = [F.normalize(c_cache[k], dim=-1) for k in corpus]
    c_lengths = [s.shape[0] for s in c_slots]
    c_all = torch.cat(c_slots).to(device)                         # [S_c, D]
    N_q, N_c = len(queries), len(corpus)
    c_sidx = torch.repeat_interleave(
        torch.arange(N_c, device=device), torch.tensor(c_lengths, device=device)
    )                                                              # [S_c]

    rules = ["h_min_to_p"]
    if all_rules:
        rules += ["maxsim", "pool"]
        c_counts = torch.tensor(c_lengths, dtype=torch.float, device=device)
        c_pool = F.normalize(torch.stack([c_cache[k].mean(0) for k in corpus]), dim=-1).to(device)
    ranks = {r: np.zeros(N_q, dtype=np.int64) for r in rules}

    for start in tqdm(range(0, N_q, chunk_size), desc=f"{TAG} scoring"):
        q_slots = [F.normalize(q_cache[k], dim=-1) for k in queries[start : start + chunk_size]]
        B = len(q_slots)
        q_lens = [s.shape[0] for s in q_slots]
        q = torch.cat(q_slots).to(device)                          # [T, D]
        q_bidx = torch.repeat_interleave(
            torch.arange(B, device=device), torch.tensor(q_lens, device=device)
        )                                                          # [T]

        sim = q @ c_all.T                                          # [T, S_c]
        q2c = torch.full((q.shape[0], N_c), float("-inf"), device=device)
        q2c.scatter_reduce_(1, c_sidx.unsqueeze(0).expand(q.shape[0], -1), sim, reduce="amax")

        hmin = torch.full((B, N_c), float("inf"), device=device)
        hmin.scatter_reduce_(0, q_bidx.unsqueeze(1).expand(-1, N_c), q2c, reduce="amin")
        mats = {"h_min_to_p": hmin}

        if all_rules:
            q_counts = torch.tensor(q_lens, dtype=torch.float, device=device)
            q_pool_sum = torch.zeros(B, q.shape[1], device=device)
            q_pool_sum.index_add_(0, q_bidx, q)
            q_pool = F.normalize(q_pool_sum / q_counts.unsqueeze(1), dim=-1)
            mats["pool"] = q_pool @ c_pool.T

            fwd_sum = torch.zeros(B, N_c, device=device)
            fwd_sum.index_add_(0, q_bidx, q2c)
            fwd = fwd_sum / q_counts.unsqueeze(1)
            max_per_c_slot = torch.full((B, sim.shape[1]), float("-inf"), device=device)
            max_per_c_slot.scatter_reduce_(
                0, q_bidx.unsqueeze(1).expand(-1, sim.shape[1]), sim, reduce="amax"
            )
            bwd_sum = torch.zeros(B, N_c, device=device)
            bwd_sum.scatter_add_(1, c_sidx.unsqueeze(0).expand(B, -1), max_per_c_slot)
            mats["maxsim"] = 0.5 * (fwd + bwd_sum / c_counts)

        g = gold[start : start + B]
        for rule, mat in mats.items():
            ranks[rule][start : start + B] = _rank(mat, g).cpu().numpy()

    return ranks


def _pooled_ranks(
    q_cache: Dict[str, torch.Tensor],
    queries: List[str],
    c_cache: Dict[str, torch.Tensor],
    corpus: List[str],
    gold: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """
    Gold ranks under cosine similarity of single-vector embeddings.
    """
    q = torch.cat([q_cache[k] for k in queries]).to(device)       # [N_q, D]
    c = torch.cat([c_cache[k] for k in corpus]).to(device)        # [N_c, D]
    ranks = np.zeros(len(queries), dtype=np.int64)
    for start in range(0, len(queries), chunk_size):
        scores = q[start : start + chunk_size] @ c.T
        ranks[start : start + chunk_size] = _rank(scores, gold[start : start + chunk_size]).cpu().numpy()
    return ranks


def _retrieval_metrics(ranks: np.ndarray, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    res = {f"R@{k}": float((ranks <= k).mean() * 100) for k in ks}
    res["MRR"] = float((1.0 / ranks).mean())
    return res


def _print_section(all_metrics: Dict[str, Dict[str, Dict[str, float]]], section: str, title: str) -> None:
    hdr = f"  {'encoder':20s}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'MRR':>7}"
    print(f"\n{title}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, sections in all_metrics.items():
        m = sections.get(section)
        if m is None:
            continue
        print(
            f"  {name:20s}  {m['R@1']:6.2f}%  {m['R@5']:6.2f}%"
            f"  {m['R@10']:6.2f}%  {m['MRR']:.4f}"
        )


def run(
    encoders: Sequence,
    device: torch.device,
    repo_id: str = REPO_ID,
    all_rules: bool = False,
    chunk_size: int = 128,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Runs PropSearch for every encoder. Returns {scorer: {section: {metric: value}}}.
    If `all_rules`, APE is also scored with maxsim and pool.
    """
    data = load_propsearch(repo_id)
    print(
        f"{TAG} corpus={len(data.corpus)} (source={data.n_sources}, "
        f"distractors={len(data.corpus) - data.n_sources})  queries={len(data.queries)}"
    )
    gold = torch.tensor(data.gold, dtype=torch.long, device=device)

    # Strata by source-sentence atom count: 1, 2, 3, 4, 5, >5
    n_atoms = np.array(data.source_n_atoms)
    strata = [(f"src_natoms={n}", n_atoms == n) for n in range(1, 6)]
    strata.append(("src_natoms>5", n_atoms > 5))
    strata = [(name, mask) for name, mask in strata if mask.any()]

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for enc in encoders:
        print(f"{TAG} encoding {enc.name} ...")
        c_cache = enc.encode(data.corpus)
        q_cache = enc.encode(data.queries)
        if enc.multi_vector:
            ranks = _multi_vector_ranks(
                q_cache, data.queries, c_cache, data.corpus, gold, device, chunk_size, all_rules
            )
            ranks = {f"{enc.name}_{rule}": r for rule, r in ranks.items()}
        else:
            ranks = {enc.name: _pooled_ranks(q_cache, data.queries, c_cache, data.corpus, gold, device, chunk_size)}
        del c_cache, q_cache

        for name, r in ranks.items():
            all_metrics[name] = {"Overall": _retrieval_metrics(r)}
            for stratum, mask in strata:
                all_metrics[name][stratum] = _retrieval_metrics(r[mask])
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _print_section(all_metrics, "Overall", f"{TAG} Overall (n={len(data.queries)})")
    for stratum, mask in strata:
        _print_section(all_metrics, stratum, f"{TAG} {stratum} (n={int(mask.sum())})")
    return all_metrics
