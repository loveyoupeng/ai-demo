from __future__ import annotations

import pickle
import os
from typing import Union

from model.transformer import Transformer
from model.pytorch.transformer import PyTorchTransformer
from tokenizer.char_tokenizer import CharTokenizer


class ModelCheckpoint:
    """
    Handles saving and loading of Transformer models and tokenizers.

    Supports both NumPy and PyTorch backends via duck typing:
    both models implement ``get_params()``, ``set_params()``, ``embed_dim``,
    ``num_layers``, ``blocks``, and ``max_seq_len``.
    """

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_checkpoint(
        self,
        model: Union[Transformer, PyTorchTransformer],
        tokenizer: CharTokenizer,
        filename: str,
    ) -> str:
        """Save model checkpoint to disk."""
        path = os.path.join(self.base_dir, f"{filename}.pkl")
        checkpoint = {
            "model_params": model.get_params(),
            "tokenizer": tokenizer,
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": model.embed_dim,
            "num_layers": model.num_layers,
        }
        # Only save blocks info if available at all — required for loading later
        if hasattr(model, "blocks") and model.blocks and len(model.blocks) > 0:
            block0 = model.blocks[0]
            if hasattr(block0, "mha"):
                checkpoint["num_heads"] = block0.mha.num_heads  # type: ignore[union-attr]
            if hasattr(block0, "moe"):
                checkpoint["num_experts"] = block0.moe.num_experts  # type: ignore[union-attr]
        if hasattr(model, "max_seq_len"):
            checkpoint["max_seq_len"] = model.max_seq_len

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(
        self,
        filename: str,
        model_class: Union[type[Transformer], type[PyTorchTransformer]],
        tokenizer_class: type[CharTokenizer],
    ) -> tuple[object, object]:
        """Load model and tokenizer from a saved checkpoint on disk."""
        filepath = os.path.join(self.base_dir, f"{filename}.pkl")
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)  # type: ignore[assignment]

        # 1. Reconstruct Tokenizer
        tokenizer = tokenizer_class()
        tokenizer.chars = checkpoint["tokenizer"].chars
        tokenizer.vocab_size = checkpoint["tokenizer"].vocab_size
        tokenizer.char_to_int = checkpoint["tokenizer"].char_to_int
        tokenizer.int_to_char = checkpoint["tokenizer"].int_to_char

        # 2. Reconstruct Model (instantiated by caller to control backend)
        model = model_class(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            num_layers=checkpoint["num_layers"],
            num_heads=checkpoint["num_heads"],
            num_experts=checkpoint["num_experts"],
            max_seq_len=checkpoint["max_seq_len"],
        )
        model.set_params(checkpoint["model_params"])

        print(f"Checkpoint loaded from {filepath}")
        return model, tokenizer
