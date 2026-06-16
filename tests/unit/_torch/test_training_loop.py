"""C9.1: Tests for PyTorch Training Loop.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch
import torch.nn.functional as F


class TestTrainingLoop:
    """Test the training loop orchestration."""

    def test_training_reduces_loss(self) -> None:
        """Training over several steps should reduce the loss."""

        from impl._torch.training import train_step

        # Create a tiny model — a single Linear layer for quick convergence
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        torch.manual_seed(42)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Use fixed input/target pairs so model can learn
        batch_input = torch.randn(8, 8)
        batch_target = torch.randint(0, 8, (8,))

        loss_history: list[float] = []
        for _ in range(50):
            loss = train_step(model, batch_input, batch_target, optimizer, loss_fn)
            loss_history.append(loss)

        # Loss should decrease over time
        first_quarter = sum(loss_history[:5]) / 5
        last_quarter = sum(loss_history[45:]) / 5
        assert last_quarter < first_quarter, f"Loss should decrease: first={first_quarter:.3f}, last={last_quarter:.3f}"

    def test_params_update(self) -> None:
        """Model parameters change after training steps."""
        from impl._torch.training import train_step

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        torch.manual_seed(42)
        model = TinyModel()
        initial_weight = model.fc.weight.data.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(10):
            batch_input = torch.randn(2, 4)
            batch_target = torch.randint(0, 4, (2,))
            train_step(model, batch_input, batch_target, optimizer, loss_fn)

        assert not torch.allclose(model.fc.weight.data, initial_weight), "Weights should change after training"

    def test_gradient_existence(self) -> None:
        """After backward, model parameters have valid gradients before zero_grad."""
        # Create model with requires_grad
        fc = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randint(0, 4, (2,))

        logits = fc(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        # Gradients should be accumulated on parameters
        assert fc.weight.grad is not None, "No gradient for fc.weight"
        assert not torch.all(fc.weight.grad == 0), "All-zero gradient for fc.weight"
        assert fc.bias.grad is not None, "No gradient for fc.bias"
        assert not torch.all(fc.bias.grad == 0), "All-zero gradient for fc.bias"
