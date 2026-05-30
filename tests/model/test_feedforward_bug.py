import pytest
import numpy as np
from model.layers import FeedForward

def numerical_gradient(ffn, x, eps=1e-6):
    grad_x = np.zeros_like(x)
    output_orig, _ = ffn.forward(x), None # forward doesn't return anything extra in current implementation but we need to be careful
    # Actually FeedForward.forward returns output
    
    # We need to define a scalar loss to compute numerical gradient
    # Let's say L = sum(ffn.forward(x))
    
    def get_loss(x_val):
        return np.sum(ffn.forward(x_val))

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + eps
        loss_plus = get_loss(x)
        
        x[idx] = old_val - eps
        loss_minus = get_loss(x)
        
        grad_x[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        x[idx] = old_val
        it.iternext()
    return grad_x

@pytest.mark.timeout(10)
def test_feedforward_backward_numerical():
    batch_size = 2
    seq_len = 3
    embed_dim = 4
    dim_ff = 8
    
    ffn = FeedForward(embed_dim, dim_ff)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = ffn.forward(x)
    
    # Dummy gradient for loss w.r.t output
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Analytical gradient
    grad_x_analytical = ffn.backward(grad_output)
    
    # Numerical gradient
    # L = sum(output * grad_output)
    def get_loss(x_val):
        out = ffn.forward(x_val)
        return np.sum(out * grad_output)

    grad_x_numerical = np.zeros_like(x)
    eps = 1e-6
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + eps
        loss_plus = get_loss(x)
        
        x[idx] = old_val - eps
        loss_minus = get_loss(x)
        
        grad_x_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        x[idx] = old_val
        it.iternext()

    np.testing.assert_allclose(grad_x_analytical, grad_x_numerical, rtol=1e-4, atol=1e-4)

@pytest.mark.timeout(10)
def test_feedforward_params_numerical():
    batch_size = 2
    seq_len = 3
    embed_dim = 4
    dim_ff = 8
    
    ffn = FeedForward(embed_dim, dim_ff)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    output = ffn.forward(x)
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    ffn.backward(grad_output)
    
    grads = ffn.get_grads()
    params = ffn.get_params()
    
    for key in params:
        param = params[key]
        grad_param = grads[key]
        
        def get_loss(p_val):
            # Temporarily replace param with p_val
            old_val = param.copy()
            # We need to be careful how we replace it because it's an attribute
            # Since they are numpy arrays, we can do:
            if key == "W1": ffn.W1 = p_val
            elif key == "b1": ffn.b1 = p_val
            elif key == "W2": ffn.W2 = p_val
            elif key == "b2": ffn.b2 = p_val
            
            res = np.sum(ffn.forward(x) * grad_output)
            
            # Restore
            if key == "W1": ffn.W1 = old_val
            elif key == "b1": ffn.b1 = old_val
            elif key == "W2": ffn.W2 = old_val
            elif key == "b2": ffn.b2 = old_val
            return res

        grad_param_numerical = np.zeros_like(param)
        eps = 1e-6
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            param[idx] = old_val + eps
            loss_plus = get_loss(param)
            
            param[idx] = old_val - eps
            loss_minus = get_loss(param)
            
            grad_param_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            param[idx] = old_val
            it.iternext()
            
        np.testing.assert_allclose(grad_param, grad_param_numerical, rtol=1e-4, atol=1e-4)
