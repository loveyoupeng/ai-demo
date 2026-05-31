import time
import numpy as np
from typing import Dict
from utils.backend_interface import BaseTransformerBackend


class Profiler:
    """
    Tool to profile the performance of a Transformer backend.
    Measures latency, throughput, and peak memory presence.
    """

    def __init__(self, backend: BaseTransformerBackend):
        self.backend = backend

    def profile_latency(
        self, input_ids: np.ndarray, num_runs: int = 10, mode: str = "forward"
    ) -> Dict[str, float]:
        """
        Measures the average latency of a single operation.

        Args:
            input_ids: Input token IDs [Batch, Seq_Len].
            num_runs: Number of iterations to average.
            mode: "forward" or "backward".

        Returns:
            A dictionary containing 'avg_latency', 'min_latency', 'max_latency'.
        """
        latencies = []

        # Warmup
        for _ in range(max(1, num_runs // 5)):
            if mode == "forward":
                logits, cache = self.backend.forward(input_ids)
            else:
                # Need a dummy logit and cache for backward
                dummy_logits = np.zeros(
                    (input_ids.shape[0], input_ids.shape[1], 1)
                )  # Simplified
                # In a real scenario, we'd need consistent vocab size, but for profiling we can mock
                # For now, let's assume backward is called on whatever the forward produced
                _, cache = self.backend.forward(input_ids)
                self.backend.backward(dummy_logits, cache)

        # Measurement
        for _ in range(num_runs):
            start_time = time.perf_counter()
            if mode == "forward":
                logits, cache = self.backend.forward(input_ids)
                _ = logits  # prevent optimization
            else:
                # For backward, we need logits and cache from a forward pass
                logits, cache = self.backend.forward(input_ids)
                # Mocking grad_logits shape [Batch, Seq, Vocab]
                # We'll use a simple zero tensor for profiling
                grad_logits = np.zeros_like(logits)
                self.backend.backward(grad_logits, cache)

            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

        return {
            "avg_latency": float(np.mean(latencies)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
            "std_latency": float(np.std(latencies)),
        }

    def profile_throughput(
        self, input_ids: np.ndarray, num_steps: int = 10
    ) -> Dict[str, float]:
        """
        Measures throughput in tokens per second.

        Args:
            input_ids: Input token IDs [Batch, Seq_Len].
            num_steps: Number of steps to average.

        Returns:
            Dictionary containing 'tokens_per_second'.
        """
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len * num_steps

        latencies = self.profile_latency(input_ids, num_runs=num_steps, mode="forward")[
            "avg_latency"
        ]

        tokens_per_sec = total_tokens / latencies
        return {"tokens_per_second": float(tokens_per_sec)}

    def profile_memory(self, input_ids: np.ndarray) -> Dict[str, float]:
        """
        Measures the memory usage (placeholder for now).

        Note: Real implementation would use psutil for system RAM
        or torch.cuda for VRAM.
        """
        # Placeholder: returns 0.0 for now
        return {"peak_memory_mb": 0.0}
