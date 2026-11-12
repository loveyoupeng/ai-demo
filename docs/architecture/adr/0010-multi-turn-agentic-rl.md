# ADR 0010: Multi-turn agentic RL (tool-use trajectories) as the RL stage

- Status: accepted
- Date: 2026-11-09
- Supersedes: the single-turn-only scope of ADR 0004's RL stage (the
  thinking-SFT format and the GRPO machinery are unchanged)

## Context

ADR 0004's RL stage is single-turn: one prompt → one completion → reward.
The 2026 post-training frontier moved to **multi-turn, tool-using
trajectories**: the policy emits a `<|tool_call|>`, the environment executes
it and returns `<|tool_response|>`, the policy continues, and only the final
`<|answer|>` is scored (ARPO, Turn-PPO, tree-search RL — ICLR/ACL 2026).
Turn-level advantage (per-turn credit assignment instead of one trajectory
score) is the standard fix for long-horizon credit assignment.

The repo already has every ingredient: the tool-call SFT format
(`<|tool_call|>` / `<|tool_response|>`, stage 3 of the v3 pipeline) and the
GRPO math (ADR 0004).

## Decision

`scripts/rl.py` gains a `--multi-turn` mode (the v4 pipeline's default RL
stage; `--single-turn` keeps the ADR 0004 behavior):

1. **Environment** (`shared/agentic.py`): a tiny deterministic tool sandbox —
   a `calculator` tool whose `expr` argument is parsed with `ast` and
   evaluated with a whitelist (arithmetic only, no names/calls), returning
   the result as `<|tool_response|>{...}`. Safe by construction (no eval
   of arbitrary code), deterministic (same expr → same result).
2. **Rollout**: prompt → sample completion up to `<|tool_call|>` → run the
   tool → append `<|tool_response|>` → resample continuation (up to N turns,
   default 2) → final `<|answer|>`. Each turn's tokens are recorded.
3. **Reward**: unchanged parser (`shared.thinking.reward`) on the FINAL
   `<|answer|>`: 1.0 correctness + 0.2 format, plus a small tool-use bonus
   (+0.1) when at least one well-formed tool call was made and consumed —
   the signal that keeps the agent actually using the tool.
4. **Advantage**: turn-level — the trajectory reward is assigned to every
   turn's tokens (Turn-PPO style), then group-normalized across the G
   trajectories per prompt as in GRPO. KL penalty vs the frozen reference is
   unchanged.

Data: GSM8K prompts (the calculator can solve the arithmetic sub-steps;
the synthetic two-step floor from the think stage guarantees a learnable
signal).

## Why not alternatives

- **Outcome-only reward, no tool bonus**: the model can skip the tool and
  still score on easy prompts; the small tool bonus keeps the agentic
  behavior alive at demo scale.
- **Full PPO with a learned critic**: GRPO's group-relative baseline is the
  demo-appropriate choice (ADR 0004); turn-level assignment is the only
  change, matching Turn-PPO's finding that turn-level advantages beat
  trajectory-level for multi-turn.

## Consequences

- `shared/agentic.py` owns the tool sandbox + the `<|tool_call|>` JSON parse
  + turn assembly, shared by all backends' RL loops.
- `impl/_np/rl.py` gains a multi-turn rollout/iteration (teaching path:
  explicit per-turn sampling + advantage); `scripts/rl.py --multi-turn`
  wires it for numpy and the torch family alike.
- The v3→v4 pipeline stage 5 calls `scripts.rl --multi-turn`; the
  `pipeline` config marker bumps 4 → 5 so stale checkpoints bootstrap.
- `rl_log.json` additionally records per-turn tool-call rates so the
  learning page/docs can show the agentic behavior emerging.
