"""B1.1: Embedding layer — input token lookup in weight matrix.

The Embedding module owns its (vocab_size, embed_dim) weight table; forward
performs a row lookup: input_ids (B, S) → (B, S, D).
"""

import numpy as np


class TestEmbeddingForward:
    """Test the Embedding forward pass."""

    def test_output_shape(self):
        """Embedding output shape: [batch, seq_len, embed_dim].

        Input: input_ids of shape [batch=2, seq_len=4] with
               weight matrix of shape [vocab_size=16, embed_dim=8]
        Expected output: [2, 4, 8]
        """
        from impl._np.embedding import Embedding

        input_ids = np.array([[0, 5, 10, 15], [1, 3, 7, 13]], dtype=np.int32)

        emb = Embedding(vocab_size=16, embed_dim=8, seed=0)
        output = emb.forward(input_ids)

        assert output.shape == (2, 4, 8), f"Expected (2, 4, 8), got {output.shape}"

    def test_lookup_correctness(self):
        """Verify embedding[i] maps to the i-th row of the weight table.

        For a single token, output[0,0] should equal weight[token_id].
        """
        from impl._np.embedding import Embedding

        input_ids = np.array([[5]], dtype=np.int32)

        emb = Embedding(vocab_size=7, embed_dim=3, seed=0)
        emb.weight[5] = [16.0, 17.0, 18.0]  # row 5

        output = emb.forward(input_ids)

        np.testing.assert_array_equal(output[0, 0, :], emb.weight[5], err_msg="embedding[5] != weight[5]")
        np.testing.assert_array_equal(output[0, 0, :], [16.0, 17.0, 18.0])

    def test_batch_handling(self):
        """Multiple sequences processed in parallel with correct lookups.

        Each position in the batch should independently look up the correct
        embedding row for its token ID.
        """
        from impl._np.embedding import Embedding

        input_ids = np.array(
            [[0, 1], [2, 3]],
            dtype=np.int32,
        )

        emb = Embedding(vocab_size=4, embed_dim=3, seed=0)
        for i in range(4):
            emb.weight[i, :] = float(i)  # row i = [i, i, i]

        output = emb.forward(input_ids)

        # Input [[0, 1], [2, 3]] → output[0,0]=[0,0,0], output[0,1]=[1,1,1]
        #                output[1,0]=[2,2,2], output[1,1]=[3,3,3]
        np.testing.assert_array_equal(output[0, 0, :], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(output[0, 1, :], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(output[1, 0, :], [2.0, 2.0, 2.0])
        np.testing.assert_array_equal(output[1, 1, :], [3.0, 3.0, 3.0])
