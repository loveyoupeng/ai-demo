import numpy as np
from typing import Optional
from src.model.attention import MultiHeadAttention
from src.model.moe import MoELayer
from src.model.layers import LayerNorm


class TransformerBlock(object):
    """
    A single Transformer Decoder Block.
    Combines Multi-Head Attention (MHA) and a Mixture of Experts (MoE) layer,
    wrapped in residual connections and Layer Normalization.
    """

    def __init__(self, embed_dim: int, mha: MultiHeadAttention, moe: MoELayer):
        self.mha = mha
        self.moe = moe

        # Pre-norm architecture: LayerNorm is applied BEFORE the sub-layers
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len]
        Returns:
            [Batch, Seq_Len, Embed_Dim]
        """
        # 1. Self-Attention Sub-layer (Pre-Norm)
        # x = x + MHA(LN(x))
        residual = x
        x = self.ln1.forward(x)
        x = self.mha.forward(x, mask=mask)
        x = residual + x

        # 2. Feed-Forward / MoE Sub-layer (Pre-Norm)
        # x = x + MoE(LN(x))
        residual = x
        x = self.ln2.forward(x)
        x = self.moe.forward(x)
        x = residual + x

        return x


class Transformer:
    """
    The full Decoder-only Transformer model.
    Composed of a stack of Transformer blocks, token/positional embeddings,
    and a language model head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        max_seq_len: int = 512,
    ):

        from src.model.layers import TokenEmbedding, PositionalEmbedding

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 1. Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEmbedding(max_seq_len, embed_dim)

        # 2. Transformer Stack
        self.blocks = []
        for _ in range(num_layers):
            mha = MultiHeadAttention(embed_dim, num_heads)
            moe = MoELayer(embed_dim, num_experts)
            self.blocks.append(TransformerBlock(embed_dim, mha, moe))

        # 3. Language Model Head (Linear projection to vocab)
        # [Embed_Dim, Vocab_Size]
        self.lm_head = np.random.randn(embed_dim, vocab_size) * 0.01

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            input_ids: [Batch, Seq_Len] integer token IDs
        Returns:
            logits: [Batch, Seq_Len, Vocab_Size]
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token + Positional Embeddings
        # [Batch, Seq_Len, Embed_Dim]
        x = self.token_embedding.forward(input_ids)
        pos_pe = self.pos_embedding.forward()  # [Max_Seq_Len, Embed_Dim]
        x = x + pos_pe[:seq_len, :]

        # 2. Causal Mask
        # [Seq_Len, Seq_Len]
        mask = np.tril(np.ones((seq_len, seq_len)))

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block.forward(x, mask=mask)

        # 4. LM Head (Logits)
        # [Batch, Seq_Len, Vocab_Size]
        logits = np.dot(x, self.lm_head)

        return logits
