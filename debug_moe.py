import numpy as np
from model.moe import MoELayer


def debug():
    batch_size = 2
    seq_len = 3
    embed_dim = 4
    num_experts = 4
    k = 2

    moe = MoELayer(embed_dim, num_experts, num_experts_per_token=k)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass to populate cache
    output, cache = moe.forward(x)

    # Dummy gradient for loss w.r.t output
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)

    # Analytical gradient
    grad_x_analytical = moe.backward(x, grad_output, cache)
    print(f"grad_x_analytical type: {type(grad_x_analytical)}")
    if isinstance(grad_x_analytical, tuple):
        print(f"grad_x_analytical tuple length: {len(grad_x_analytical)}")
        print(f"grad_x_analytical[0] type: {type(grad_x_analytical[0])}")
        print(f"grad_x_analytical[0] shape: {grad_x_analytical[0].shape}")
    else:
        print(f"grad_x_analytical shape: {grad_x_analytical.shape}")

    # Numerical gradient
    def get_loss(x_val):
        out, _ = moe.forward(x_val)
        return np.sum(out * grad_output)

    grad_x_numerical = np.zeros_like(x)
    eps = 1e-6
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
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

    print(f"grad_x_numerical type: {type(grad_x_numerical)}")
    print(f"grad_x_numerical shape: {grad_x_numerical.shape}")

    try:
        if isinstance(grad_x_analytical, tuple):
            grad_x_analytical = grad_x_analytical[0]
        np.testing.assert_allclose(
            grad_x_analytical, grad_x_numerical, rtol=1e-4, atol=1e-4
        )
        print("Assertion passed!")
    except Exception as e:
        print(f"Assertion failed: {e}")


if __name__ == "__main__":
    debug()
