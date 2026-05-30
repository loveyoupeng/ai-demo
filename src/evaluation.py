import numpy as np


def calculate_perplexity(loss: float) -> float:
    """
    Calculates perplexity from the cross-entropy loss.
    Perplexity is defined as the exponent of the loss: exp(loss).
    It measures how well the probability distribution predicts the sample.

    Args:
        loss: The cross-entropy loss (scalar).

    Returns:
        The perplexity value.
    """
    return np.exp(loss)


def calculate_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculates the categorical accuracy.

    Args:
        logits: Model outputs of shape [Batch, Seq_Len, Vocab_Size]
        targets: Ground truth token IDs of shape [Batch, Seq_Len]

    Returns:
        The accuracy as a fraction between 0 and 1.
    """
    # Get the predicted token IDs (indices of the maximum logit)
    # pred_ids shape: [Batch, Seq_Len]
    pred_ids = np.argmax(logits, axis=-1)

    # Compare predictions with targets
    # matches shape: [Batch, Seq_Len]
    matches = pred_ids == targets

    # Return mean of matches
    return float(np.mean(matches))
