import pytest
import numpy as np
from src.model.moe import Router, Expert, MoELayer

@pytest.mark.timeout(5)
def test_router_output_shape():
    """
    Test that the router produces a valid probability distribution over experts.
    """
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    num_experts = 8
    
    router = Router(embed_dim, num_experts)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    scores = router.forward(x)
    
    # Shape should be [Batch, Seq_Len, Num_Experts]
    assert scores.shape == (batch_size, seq_len, num_experts)
    # Scores should sum to 1 (softmax property)
    assert np.allclose(np.sum(scores, axis=-1), 1.0, atol=1e-5)
    # Scores should be non-negative
    assert np.all(scores >= 0)

@pytest.mark.timeout(5)
def test_expert_shape():
    """
    Test that an expert (FFN) maintains input shape.
    """
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    dim_ff = 64
    
    expert = Expert(embed_dim, dim_ff)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    output = expert.forward(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)

@pytest.mark.timeout(5)
def test_moe_layer_shape():
    """
    Test that the MoE layer produces the correct output shape.
    """
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    num_experts = 8
    
    moe = MoELayer(embed_dim, num_experts, num_experts_per_token=2)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    output = moe.forward(x)
    
    # Output shape should match input shape
    assert output.shape == (batch_size, seq_len, embed_dim)

@pytest.mark.timeout(5)
def test_moe_top_k_routing_logic():
    """
    Test that MoE uses Top-K experts.
    We can check this by ensuring the output is a weighted sum of specifically chosen experts.
    """
    batch_size = 1
    seq_len = 1
    embed_dim = 4
    num_experts = 4
    k = 2
    
    moe = MoELayer(embed_dim, num_experts, num_experts_per_token=k)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Manual calculation of expected output
    scores = moe.router.forward(x) # [1, 1, 4]
    
    # Get top k indices and weights correctly
    idx = np.argsort(scores, axis=-1)[..., -k:]
    val = np.take_along_axis(scores, idx, axis=-1)
    # Normalize weights for the top-k
    val = val / (np.sum(val, axis=-1, keepdims=True) + 1e-8)
    
    # Get individual expert outputs
    expert_outputs = []
    for i in range(num_experts):
        expert_outputs.append(moe.experts[i].forward(x))
    expert_outputs = np.array(expert_outputs) # [Num_Experts, Batch, Seq, Dim]
    
    # Expected output: Sum_{i in top_k} weight_i * expert_i(x)
    expected_output = np.zeros_like(x)
    for i in range(k):
        expert_idx = idx[0, 0, i]
        weight = val[0, 0, i]
        expected_output[0, 0, :] += weight * expert_outputs[expert_idx, 0, 0, :]
        
    output = moe.forward(x)
    np.testing.assert_allclose(output, expected_output, atol=1e-5)
