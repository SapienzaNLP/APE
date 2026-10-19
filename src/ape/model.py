"""
Atomic Proposition Embeddings (APE) - model architecture and Hugging Face Hub integration.
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class APEConfig:
    """
    Architecture of an APE model, along with its inference defaults.
    Serialized to `config.json` when the model is saved or pushed to the Hub.
    """
    encoder_name: str = "BAAI/bge-small-en-v1.5"
    num_slots: int = 8
    decoder_ff: int = 1536
    decoder_heads: int = 8
    decoder_layers: int = 3
    decoder_dropout: float = 0.1

    # Inference defaults
    activity_threshold: float = 0.6
    max_seq_len: int = 128


class APE(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="ape",
    tags=["atomic-propositions", "multi-vector", "sentence-embeddings"],
):
    """
    Backbone encoder + slot decoder + embedding / activity heads.

    Each of the `num_slots` learned queries attends to the encoder's token states
    and yields one proposition embedding plus an activity logit. At inference,
    the active slots (activity > threshold) are the sentence's atomic propositions.

    Load a trained model from the Hugging Face Hub or a local directory with
    `APE.from_pretrained(...)`; save it with `save_pretrained` / `push_to_hub`.
    """

    def __init__(self, config: APEConfig | None = None):
        super().__init__()
        self.config = config or APEConfig()

        # Architecture only, as trained weights are loaded afterwards, either by
        # `from_pretrained` (full APE) or by the training script (backbone).
        encoder_config = AutoConfig.from_pretrained(self.config.encoder_name)
        self.encoder = AutoModel.from_config(encoder_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        self.encoder_frozen: bool = False

        embed_dim = encoder_config.hidden_size
        self.slot_queries = nn.Parameter(torch.randn(self.config.num_slots, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=self.config.decoder_heads,
            dim_feedforward=self.config.decoder_ff,
            dropout=self.config.decoder_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.decoder_layers)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.embed_head = nn.Linear(embed_dim, embed_dim)
        self.act_head = nn.Linear(embed_dim, 1)

    def freeze_backbone(self) -> None:
        """
        Freeze encoder backbone.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze encoder backbone.
        """
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder_frozen = False

    def forward(
        self,
        sentence_input_ids: torch.Tensor,
        sentence_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns slot embeddings [B, N_Slots, D] and activity logits [B, N_Slots].
        """
        token_states = self.encoder(
            input_ids=sentence_input_ids, attention_mask=sentence_attention_mask
        ).last_hidden_state
        B = token_states.size(0)

        memory_key_padding_mask = sentence_attention_mask == 0
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)

        decoded = self.decoder(
            tgt=queries,
            memory=token_states,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoded = self.decoder_norm(decoded)

        slot_embeds = self.embed_head(decoded)
        slot_act_logits = self.act_head(decoded).squeeze(-1)
        return slot_embeds, slot_act_logits

    def _batches(
        self,
        sentences: List[str],
        batch_size: int,
        max_seq_len: int | None,
        show_progress_bar: bool,
    ) -> Iterator[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        """
        Runs the model over length-sorted batches (less padding).
        Yields (sentence indices, slot embeddings [B, N_Slots, D], activity probs [B, N_Slots]).
        """
        device = next(self.parameters()).device
        max_seq_len = max_seq_len or self.config.max_seq_len
        order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))

        starts = range(0, len(order), batch_size)
        for start in tqdm(starts, desc="APE encode", disable=not show_progress_bar):
            idx = order[start : start + batch_size]
            enc = self.tokenizer(
                [sentences[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(device)
            slot_embeds, slot_act_logits = self(enc["input_ids"], enc["attention_mask"])
            yield idx, slot_embeds.float(), torch.sigmoid(slot_act_logits.float())

    @torch.no_grad()
    def encode(
        self,
        sentences: str | List[str],
        batch_size: int = 64,
        activity_threshold: float | None = None,
        max_seq_len: int | None = None,
        normalize_embeddings: bool = False,
        return_scores: bool = False,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = False,
    ):
        """
        Encode each sentence into the set of its atomic proposition embeddings.

        Args:
            sentences: The input sentence(s) to encode.
            batch_size: Batch size used for computation.
            activity_threshold: Slots whose activity score exceeds this threshold are
                returned. Defaults to `config.activity_threshold`. If no slot is active,
                the most active one is returned, so every sentence has >= 1 proposition.
            max_seq_len: Truncation length. Defaults to `config.max_seq_len`.
            normalize_embeddings: Whether to L2-normalize the returned embeddings.
            return_scores: Whether to also return the activity score of each embedding.
            convert_to_numpy: Whether to return numpy arrays instead of (CPU) tensors.
            show_progress_bar: Whether to show a progress bar.

        Returns:
            For a list of sentences, a list with one `[n_i, D]` embedding matrix per
            sentence (`n_i` = number of propositions found in sentence i). If
            `return_scores`, a tuple `(embeddings, scores)` where `scores[i]` has shape
            `[n_i]`. For a single string, the entries for that sentence only.
        """
        single_input = isinstance(sentences, str)
        sentences = [sentences] if single_input else list(sentences)
        threshold = self.config.activity_threshold if activity_threshold is None else activity_threshold

        was_training = self.training
        self.eval()

        all_embeds: List = [None] * len(sentences)
        all_scores: List = [None] * len(sentences)
        for idx, slot_embeds, slot_probs in self._batches(
            sentences, batch_size, max_seq_len, show_progress_bar
        ):
            if normalize_embeddings:
                slot_embeds = F.normalize(slot_embeds, dim=-1)

            # APE always keeps at least the most active slot.
            active = slot_probs > threshold
            rows = torch.arange(len(idx), device=active.device)
            top = slot_probs.argmax(dim=-1)
            active[rows, top] |= ~active.any(dim=-1)

            slot_embeds, slot_probs, active = slot_embeds.cpu(), slot_probs.cpu(), active.cpu()
            for j, i in enumerate(idx):
                all_embeds[i] = slot_embeds[j][active[j]]
                all_scores[i] = slot_probs[j][active[j]]

        self.train(was_training)

        if convert_to_numpy:
            all_embeds = [e.numpy() for e in all_embeds]
            all_scores = [s.numpy() for s in all_scores]
        if single_input:
            all_embeds, all_scores = all_embeds[0], all_scores[0]
        return (all_embeds, all_scores) if return_scores else all_embeds

    @torch.no_grad()
    def activity_scores(
        self,
        sentences: List[str],
        batch_size: int = 64,
        max_seq_len: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Activity score of every slot, [N_Sentences, N_Slots]; used for proposition counting.
        """
        was_training = self.training
        self.eval()

        scores = np.zeros((len(sentences), self.config.num_slots), dtype=np.float32)
        for idx, _, slot_probs in self._batches(sentences, batch_size, max_seq_len, show_progress_bar):
            scores[idx] = slot_probs.cpu().numpy()

        self.train(was_training)
        return scores
