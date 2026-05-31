import pickle
from typing import Any, Tuple
import os


class ModelCheckpoint:
    """
    Handles saving and loading of Transformer models and tokenizers.
    """

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_checkpoint(self, model: Any, tokenizer: Any, filename: str):
        """
        Saves the model parameters and the tokenizer.
        """
        checkpoint = {
            "model_params": model.get_params(),
            "tokenizer": tokenizer,
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": model.embed_dim,
            "num_layers": model.num_layers,
            "num_heads": model.blocks[0].mha.num_heads,
            "num_experts": model.blocks[0].moe.num_experts,
            "max_seq_len": model.pos_embedding.max_seq_len,
        }
        filepath = os.path.join(self.base_dir, f"{filename}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(
        self, filename: str, model_class: type, tokenizer_class: type
    ) -> Tuple[Any, Any]:
        """
        Loads a model and tokenizer from a checkpoint.
        """
        filepath = os.path.join(self.base_dir, f"{filename}.pkl")
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)

        # 1. Reconstruct Tokenizer
        tokenizer = tokenizer_class()
        tokenizer.chars = checkpoint["tokenizer"].chars
        tokenizer.vocab_size = checkpoint["tokenizer"].vocab_size
        tokenizer.char_to_int = checkpoint["tokenizer"].char_to_int
        tokenizer.int_to_char = checkpoint["tokenizer"].int_to_char

        # 2. Reconstruct Model
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
