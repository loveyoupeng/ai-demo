"""E6: Mixture of Experts (MoE) kernel with top-k routing.

Architecture:
    x[B,S,D] -> router(x) -> routing_weights[B,S,E] (softmax, top-k)
    For each expert i:
        out_i = SiLU(x @ W1[i]) * (x @ W3[i]) @ W2[i]  [B,S,D]
    Final:  sum(routing_weights[:, :, i:i+1] * out_i for i in experts)

Computational graph:
    1. Router: x @ W_router + bias -> (B,S,E)  -- softmax -> (B,S,E)
    2. Top-k:  zero out non-top-k weights -> (B,S,E)
    3. Expert: compute SwiGLU for ALL experts x ALL tokens in parallel
    4. Aggregate: sum(routing_weights[i] * expert_outs[i]) -> (B,S,D)

Memory layout:
    - routing_weights:  (B, S, E) -- per-token expert gating
    - W1, W3:           (E, D, ff_dim) -- per-expert input projections
    - W2:               (E, ff_dim, D) -- per-expert output projection
    - expert_outputs:   (E, B, S, D) -- per-expert, per-token output
"""

import torch

_MIN_KERNEL_DIM = 16


def _compute_routing_weights(
    x: torch.Tensor,
    W_router: torch.Tensor,
    bias: torch.Tensor,
    k: int = 2,
) -> torch.Tensor:
    """Compute routing weights: softmax over router scores with top-k selection.

    Parameters
    ----------
    x : torch.Tensor, shape (batch, seq_len, embed_dim)
        Input activations.
    W_router : torch.Tensor, shape (embed_dim, n_experts)
        Router projection weights.
    bias : torch.Tensor, shape (n_experts,)
        Router bias.
    k : int
        Number of top experts to select per token.

    Returns
    -------
    torch.Tensor, shape (batch, seq_len, n_experts)
        Normalized routing weights with exactly k non-zero entries per token
        when k < n_experts.
    """
    router_scores = x @ W_router + bias  # [B, S, E]
    # Stable softmax: subtract max, then softmax
    router_scores_max = router_scores.max(dim=-1, keepdim=True).values  # [B, S, 1]
    exp_scores = torch.exp(router_scores - router_scores_max)
    routing_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)  # [B, S, E]

    # ---- Top-k selection ----
    n_experts = W_router.shape[1]
    if k < n_experts:
        top_k_values, _ = torch.topk(routing_weights, k, dim=-1)  # [B, S, k]
        threshold = top_k_values.min(dim=-1, keepdim=True).values  # [B, S, 1]
        routing_weights = routing_weights * (routing_weights >= threshold).float()
        renorm_sum = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, S, 1]
        routing_weights = routing_weights / renorm_sum

    return routing_weights


def _top_k_routing(
    routing_weights: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Apply top-k selection and renormalization to routing weights.

    Parameters
    ----------
    routing_weights : torch.Tensor, shape (batch, seq_len, n_experts)
        Softmax routing weights.
    k : int
        Number of top experts to select per token.

    Returns
    -------
    torch.Tensor, shape (batch, seq_len, n_experts)
        Top-k selected and renormalized routing weights.
    """
    n_experts = routing_weights.shape[-1]
    if k < n_experts:
        top_k_values, _ = torch.topk(routing_weights, k, dim=-1)  # [B, S, k]
        threshold = top_k_values.min(dim=-1, keepdim=True).values  # [B, S, 1]
        routing_weights = routing_weights * (routing_weights >= threshold).float()
        renorm_sum = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, S, 1]
        routing_weights = routing_weights / renorm_sum
    return routing_weights


def mixture_of_experts(
    x: torch.Tensor,
    W_router: torch.Tensor,
    bias: torch.Tensor,
    W1: torch.Tensor,
    W3: torch.Tensor,
    W2: torch.Tensor,
    k: int = 2,
) -> torch.Tensor:
    """Mixture of Experts with top-k expert selection.

    Parameters
    ----------
    x : torch.Tensor, shape (batch, seq_len, embed_dim)
        Input activations.
    W_router : torch.Tensor, shape (embed_dim, n_experts)
        Router projection weights.
    bias : torch.Tensor, shape (n_experts,)
        Router bias.
    W1 : torch.Tensor, shape (n_experts, embed_dim, ff_dim)
        First projection weights for each expert.
    W3 : torch.Tensor, shape (n_experts, embed_dim, ff_dim)
        Gating projection weights for each expert.
    W2 : torch.Tensor, shape (n_experts, ff_dim, embed_dim)
        Output projection weights for each expert.
    k : int, optional
        Number of top experts to activate per token (default 2).

    Returns
    -------
    torch.Tensor, shape (batch, seq_len, embed_dim)
        Weighted sum of activated expert outputs.

    Notes
    -----
    MoE formula:
        routing_weights = softmax(x @ W_router + bias)          -> (B, S, E)
        top_k_mask = select top-k per token                      -> (B, S, E)
        For each expert i:
            expert_out_i = SiLU(x @ W1[i]) * (x @ W3[i]) @ W2[i] -> (B, S, D)
        output = sum(routing_weights_i * expert_out_i)           -> (B, S, D)
    """
    assert x.device.type == "cuda"
    W_router = W_router.to(x.dtype)
    bias = bias.to(x.dtype)
    W1 = W1.to(x.dtype)
    W3 = W3.to(x.dtype)
    W2 = W2.to(x.dtype)

    n_experts = W2.shape[0]
    B, S, D = x.shape

    routing_weights = _compute_routing_weights(x, W_router, bias, k)  # [B, S, E]

    # ---- Expert computation in batched mode ----
    # Reshape x from [B,S,D] to [B*S, D], expand to [E, B*S, D], then
    # compute SwiGLU for all experts in parallel:
    #   [E, B*S, D] @ [E, D, ff_dim] = [E, B*S, ff_dim]  (per expert)
    x_batched = x.view(B * S, D)  # [B*S, D]
    x_expanded = x_batched.unsqueeze(0).expand(n_experts, -1, -1)  # [E, B*S, D]

    # Gate: SiLU(x @ W1[i]) for each expert i
    # x_expanded: [E, B*S, D], W1: [E, D, ff_dim]
    # batched matmul: [E, B*S, D] @ [E, D, ff_dim] = [E, B*S, ff_dim]
    gate = torch.nn.functional.silu(x_expanded @ W1)  # [E, B*S, ff_dim]

    # Projection: x @ W3[i] for each expert i
    proj = x_expanded @ W3  # [E, B*S, ff_dim]

    # Gated output: gate * proj for each expert i
    gated = gate * proj  # [E, B*S, ff_dim]

    # Output projection: gated @ W2[i] for each expert i
    expert_outs = gated @ W2  # [E, B*S, D]

    # Reshape back: [E, B*S, D] -> [E, B, S, D]
    out = expert_outs.view(n_experts, B, S, D)

    # ---- Weighted sum ----
    # routing_weights: [B, S, E] -> permute -> [E, B, S, 1]
    # out: [E, B, S, D]
    # Broadcast multiply: [E, B, S, 1] * [E, B, S, D] = [E, B, S, D]
    # Sum over E: [B, S, D]
    out = (routing_weights.permute(2, 0, 1).unsqueeze(-1) * out).sum(dim=0)

    return out
