from inference import AutoregressiveGenerator


def test_generator_generation_length():
    """
    Test that the generator produces the requested number of tokens.
    """
    # Mock setup
    vocab_size = 10
    embed_dim = 8
    max_seq_len = 20

    from model.transformer import Transformer

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        max_seq_len=max_seq_len,
    )

    from tokenizer.char_tokenizer import CharTokenizer

    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")

    gen = AutoregressiveGenerator(model, tokenizer)

    prompt = "abc"
    num_new_tokens = 5

    generated_ids = gen.generate(prompt, num_new_tokens=num_new_tokens)

    # Return only the generated tokens.
    assert len(generated_ids) == num_new_tokens


def test_generator_empty_prompt():
    """
    Test generator behavior with an empty prompt.
    """
    vocab_size = 10
    embed_dim = 8
    max_seq_len = 20

    from model.transformer import Transformer

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        max_seq_len=max_seq_len,
    )

    from tokenizer.char_tokenizer import CharTokenizer

    tokenizer = CharTokenizer("abc")

    gen = AutoregressiveGenerator(model, tokenizer)

    num_new_tokens = 3
    generated_ids = gen.generate("", num_new_tokens=num_new_tokens)

    assert len(generated_ids) == num_new_tokens
