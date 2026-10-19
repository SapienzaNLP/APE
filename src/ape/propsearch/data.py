"""
Loading the PropSearch benchmark from the Hugging Face Hub.

Subsets (single "train" split each), sharing one id space ("a{i}" source sentences, "d{i}" distractors):
    source_sentences — {id, text, n_atoms}: sentences with >= 1 unambiguous atom query;
                       n_atoms is the sentence's original number of annotated atoms
    queries          — {id, atom, sentence_id}: atom -> the unique source sentence containing it
    corpus           — {id, text}: distractor sentences
    counting         — {id, text, atoms}: sentences with their full atom list (proposition counting)
"""

from dataclasses import dataclass
from typing import List, Tuple

from datasets import load_dataset

REPO_ID = "francescoortame/propsearch"


@dataclass
class PropSearch:
    """
    Retrieval task: for each query atom, retrieve its source sentence from `corpus`.
    """
    corpus: List[str]          # source sentences and distractors
    queries: List[str]         # atoms
    gold: List[int]            # index in `corpus` of each query's source sentence
    source_n_atoms: List[int]  # original atom count of each query's source sentence
    n_sources: int             # number of source sentences at the head of `corpus`


def load_propsearch(repo_id: str = REPO_ID) -> PropSearch:
    sources = load_dataset(repo_id, "source_sentences", split="train")
    queries = load_dataset(repo_id, "queries", split="train")
    distractors = load_dataset(repo_id, "corpus", split="train")

    index = {sid: i for i, sid in enumerate(sources["id"])}
    n_atoms = sources["n_atoms"]
    gold = [index[sid] for sid in queries["sentence_id"]]

    return PropSearch(
        corpus=list(sources["text"]) + list(distractors["text"]),
        queries=list(queries["atom"]),
        gold=gold,
        source_n_atoms=[n_atoms[g] for g in gold],
        n_sources=len(sources),
    )


def load_counting(repo_id: str = REPO_ID) -> Tuple[List[str], List[List[str]]]:
    """
    Returns (sentences, atoms) of the proposition-counting subset.
    """
    ds = load_dataset(repo_id, "counting", split="train")
    return list(ds["text"]), list(ds["atoms"])
