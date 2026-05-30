import numpy as np
from model.transformer import Transformer
from loss import CrossEntropyLoss
from optimizer import Adam
from training.data_loader import TextDataLoader
from tokenizer.char_tokenizer import CharTokenizer
from trainer import Trainer


def train_and_test():
    # 1. Setup Data
    text = "the quick brown fox jumps over the lazy dog. " * 10
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # 2. Hyperparameters
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    num_experts = 4
    max_seq_len = 64
    batch_size = 4
    seq_len = 16
    epochs = 2
    learning_rate = 0.001

    # 3. Initialize Model, Optimizer, Loss, Trainer
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=max_seq_len,
    )

    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    data_loader = TextDataLoader(text, tokenizer, batch_size, seq_len)

    # 4. Training
    print("Starting training...")
    trainer.fit(data_loader, epochs=epochs)
    print("Training completed.")

    # 5. Inference Test
    print("\nRunning inference test...")

    # Test string
    test_text = "the quick brown"
    input_ids = tokenizer.encode(test_text)

    # Reshape for model: [1, seq_len]
    # We need to pad or truncate to fit max_seq_len if we were using a real model,
    # but here we just need to ensure it's within max_seq_len.
    input_ids = input_ids.reshape(1, -1)

    logits, _ = model.forward(input_ids)

    # Get next token
    last_token_logits = logits[0, -1, :]
    next_token_id = np.argmax(last_token_logits)
    next_token_char = tokenizer.decode(np.array([next_token_id]))

    print(f"Input text: '{test_text}'")
    print(f"Predicted next character: '{next_token_char}'")


if __name__ == "__main__":
    train_and_test()
