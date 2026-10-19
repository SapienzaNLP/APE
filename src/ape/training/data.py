"""
Dataset module for APE training (atomic-wikisplit).
"""

import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from ape.training.config import DataConfig, TrainingConfig


def filter_dataset(
    ds: HFDataset,
    max_atoms: int,
    nli_filter: bool = True,
    nli_filter_feature_name: str = "entailed",
) -> HFDataset:
    """
    Filters a raw HF dataset by atom count and (optionally) NLI acceptance.
    Meant to run once on the full dataset, before the train/val/test split,
    so downstream split sizes are exact.

    If `nli_filter` is True but `nli_filter_feature_name` is not a column of
    `ds`, the NLI condition is skipped.
    """
    apply_nli_filter = nli_filter and nli_filter_feature_name in ds.column_names
    if nli_filter and not apply_nli_filter:
        print(
            f"[data] '{nli_filter_feature_name}' feature not found in dataset, "
            "skipping NLI filtering"
        )

    def keep(batch: dict) -> List[bool]:
        atom_condition = [
            a is not None and 1 <= len(a) <= max_atoms for a in batch["atoms"]
        ]
        if apply_nli_filter:
            nli_condition = [bool(e) for e in batch[nli_filter_feature_name]]
        else:
            nli_condition = [True] * len(atom_condition)
        return [a and n for a, n in zip(atom_condition, nli_condition)]

    return ds.filter(keep, batched=True, num_proc=4, desc="Filtering dataset...")


def load_splits(cfg: DataConfig) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """
    Loads and filters the dataset, then splits it into (train, val, test).

    Filtering (atom count + optional NLI) runs on the raw dataset before the
    split, so `val_size`/`test_size` are exact. The split is seeded by
    `data_seed`. This should guarantee identical held-out sets across runs.
    """
    print(f"[data] loading {cfg.dataset_name}")
    raw = (
        load_from_disk(cfg.dataset_name)
        if os.path.isdir(cfg.dataset_name)
        else load_dataset(cfg.dataset_name, split="train")
    )

    raw = filter_dataset(
        raw,
        max_atoms=cfg.max_atoms,
        nli_filter=cfg.nli_filter,
        nli_filter_feature_name=cfg.nli_filter_feature_name,
    )
    print(f"[data] filtered raw size: {len(raw)}")

    holdout_size = cfg.val_size + cfg.test_size
    split = raw.train_test_split(test_size=holdout_size, seed=cfg.data_seed)
    train_raw, holdout_raw = split["train"], split["test"]
    val_test_split = holdout_raw.train_test_split(test_size=cfg.test_size, seed=cfg.data_seed)
    val_raw, test_raw = val_test_split["train"], val_test_split["test"]
    print(
        f"[data] split sizes: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}"
    )
    return train_raw, val_raw, test_raw


class APEDataset(Dataset):
    """
    Wraps an (already filtered) HF dataset split. Returns dicts and collation is the collator.
    """

    def __init__(
        self,
        hf_ds_split: HFDataset,
        n_singletons: Optional[int] = None,
        seed: int = 42,
        use_paraphrased_singletons: bool = False,
    ):
        self.ds = hf_ds_split
        self.use_paraphrased_singletons = use_paraphrased_singletons

        if n_singletons is not None:
            rng = random.Random(seed)
            atoms_list = self.ds["atoms"]
            singleton_idx = [i for i, a in enumerate(atoms_list) if len(a) == 1]
            non_singleton_idx = [i for i, a in enumerate(atoms_list) if len(a) != 1]
            k = min(n_singletons, len(singleton_idx))
            sampled = rng.sample(singleton_idx, k)
            self.ds = self.ds.select(sorted(non_singleton_idx + sampled))
            print(f"[data] sampled {k}/{len(singleton_idx)} singletons")

            # If n_singletons exceeds the number of natural singletons, the deficit
            # is filled with artificial singletons: an atom (sourced from the pool
            # of extracted propositions) used as both the input sentence and the target.
            deficit = n_singletons - k
            if deficit > 0:
                if not non_singleton_idx:
                    print(
                        "[data] cannot synthesize singletons: no non-singleton examples available"
                    )
                else:
                    used: set = set()
                    synth_sentences: List[str] = []
                    attempts, max_attempts = 0, deficit * 20
                    while len(synth_sentences) < deficit and attempts < max_attempts:
                        attempts += 1
                        row_idx = rng.choice(non_singleton_idx)
                        row_atoms = atoms_list[row_idx]
                        atom_idx = rng.randrange(len(row_atoms))
                        key = (row_idx, atom_idx)
                        if key in used:
                            continue
                        used.add(key)
                        synth_sentences.append(row_atoms[atom_idx])
                    if synth_sentences:
                        synth_ds = HFDataset.from_dict(
                            {
                                "sentence": synth_sentences,
                                "atoms": [[s] for s in synth_sentences],
                            }
                        )
                        self.ds = concatenate_datasets(
                            [self.ds.select_columns(["sentence", "atoms"]), synth_ds]
                        )
                    print(
                        f"[data] synthesized {len(synth_sentences)}/{deficit} additional singletons from individual atoms"
                    )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        ex = self.ds[idx]
        atoms = ex["atoms"]
        if not self.use_paraphrased_singletons and len(atoms) == 1:
            atoms = [ex["sentence"]]
        return {"sentence": ex["sentence"], "atoms": atoms}


class APECollator:
    """
    Tokenizes sentences and atoms. Atoms are flattened with index mapping.
    """

    def __init__(
        self, tokenizer: AutoTokenizer, max_seq_len: int, max_atom_seq_len: int
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_atom_seq_len = max_atom_seq_len

    def __call__(self, batch: List[dict]) -> dict:
        sentences = [b["sentence"] for b in batch]
        atoms_per_ex = [b["atoms"] for b in batch]
        flat_atoms: List[str] = []
        atom_offsets: List[int] = [0]
        for atoms in atoms_per_ex:
            flat_atoms.extend(atoms)
            atom_offsets.append(atom_offsets[-1] + len(atoms))

        s_enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        a_enc = self.tokenizer(
            flat_atoms,
            padding=True,
            truncation=True,
            max_length=self.max_atom_seq_len,
            return_tensors="pt",
        )

        return {
            "sentence_input_ids": s_enc["input_ids"],
            "sentence_attention_mask": s_enc["attention_mask"],
            "atom_input_ids": a_enc["input_ids"],
            "atom_attention_mask": a_enc["attention_mask"],
            "atom_offsets": torch.tensor(atom_offsets, dtype=torch.long),
            "num_atoms": torch.tensor([len(a) for a in atoms_per_ex], dtype=torch.long),
        }


def build_loaders(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    encoder_name: str,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    The split and the singleton sub-sampling are seeded by `data_cfg.data_seed`;
    `seed` only controls the train DataLoader's shuffling order.
    """
    train_raw, val_raw, test_raw = load_splits(data_cfg)

    train_ds = APEDataset(
        train_raw,
        n_singletons=data_cfg.n_singletons,
        seed=data_cfg.data_seed,
        use_paraphrased_singletons=data_cfg.use_paraphrased_singletons,
    )
    val_ds = APEDataset(val_raw, use_paraphrased_singletons=data_cfg.use_paraphrased_singletons)
    test_ds = APEDataset(test_raw, use_paraphrased_singletons=data_cfg.use_paraphrased_singletons)
    print(
        f"[data] final sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    collator = APECollator(
        tokenizer,
        max_seq_len=train_cfg.max_seq_len,
        max_atom_seq_len=train_cfg.max_atom_seq_len,
    )

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    loader_kwargs = dict(collate_fn=collator, num_workers=2, pin_memory=True, drop_last=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        generator=shuffle_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg.eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.eval_batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
