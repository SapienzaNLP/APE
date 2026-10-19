"""
Converts a raw APE checkpoint (`.pt` with a "model" state dict, or a bare state
dict) into a Hub-ready directory, and optionally pushes it to the Hugging Face Hub.

Usage:
    python -m ape.training.export --checkpoint ape.pt --output_dir checkpoints/ape-small \
        [--push_to_hub org/ape-small] [--private]

Architecture flags (--encoder_name, --num_slots, ...) must match the checkpoint;
they default to `APEConfig`'s defaults.
"""

import argparse
import tempfile
from dataclasses import fields

import torch
from huggingface_hub import HfApi

from ape.model import APE, APEConfig


def export(
    checkpoint: str,
    config: APEConfig,
    output_dir: str | None = None,
    repo_id: str | None = None,
    private: bool = True,
) -> APE:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw

    model = APE(config)
    model.load_state_dict(state_dict)
    print(f"[export] loaded {checkpoint}")

    if output_dir is not None:
        model.save_pretrained(output_dir)
        print(f"[export] saved to {output_dir}")
    if repo_id is not None:
        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        # Keep an existing model card instead of overwriting it with the generated one.
        keep_card = api.file_exists(repo_id, "README.md")
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                ignore_patterns=["README.md"] if keep_card else None,
                commit_message="Upload APE model",
            )
        print(f"[export] pushed to {repo_id}")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--push_to_hub", default=None, metavar="REPO_ID")
    parser.add_argument("--private", action="store_true")
    for f in fields(APEConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = APEConfig(**{f.name: getattr(args, f.name) for f in fields(APEConfig)})
    export(args.checkpoint, config, args.output_dir, args.push_to_hub, args.private)
