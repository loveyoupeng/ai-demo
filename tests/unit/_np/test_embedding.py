"""B1.1: Embedding layer — input token lookup in weight matrix.

All tests fail initially. Implement after verifying failure.
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
        from impl._np.modules import Embedding

        input_ids = np.array([[0, 5, 10, 15], [1, 3, 7, 13]], dtype=np.int32)
        weight = np.random.default_rng(0).random((16, 8)).astype(np.float32)

        emb = Embedding()
        output = emb.forward(input_ids, weight)

        assert output.shape == (2, 4, 8), f"Expected (2, 4, 8), got {output.shape}"

    def test_lookup_correctness(self):
        """Verify embedding[i] maps to the i-th row of weight.

        For a single token, output[0,0] should equal weight[token_id].
        """
        from impl._np.modules import Embedding

        input_ids = np.array([[5]], dtype=np.int32)
        weight = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],  # row 5
                [19.0, 20.0, 21.0],
            ],
            dtype=np.float32,
        )

        emb = Embedding()
        output = emb.forward(input_ids, weight)

        np.testing.assert_array_equal(output[0, 0, :], weight[5], err_msg="embedding[5] != weight[5]")

    def test_batch_handling(self):
        """Multiple sequences processed in parallel with correct lookups.

        Each position in the batch should independently look up the correct
        embedding row for its token ID.
        """
        from impl._np.modules import Embedding

        input_ids = np.array(
            [[0, 1], [2, 3]],
            dtype=np.int32,
        )
        weight = np.zeros((4, 3), dtype=np.float32)
        for i in range(4):
            weight[i, :] = float(i)  # row i = [i, i, i]

        emb = Embedding()
        output = emb.forward(input_ids, weight)

        # Input [[0, 1], [2, 3]] → output[0,0]=[0,0,0], output[0,1]=[1,1,1]
        #                output[1,0]=[2,2,2], output[1,1]=[3,3,3]
        np.testing.assert_array_equal(output[0, 0, :], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(output[0, 1, :], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(output[1, 0, :], [2.0, 2.0, 2.0])
        np.testing.assert_array_equal(output[1, 1, :], [3.0, 3.0, 3.0])
