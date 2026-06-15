import numpy as np


def initialize_linear(d_in: int, d_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier-uniform initialization for linear layer weights.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_out : int
        Output dimension.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (d_in, d_out)
        Initialized weight matrix.

    Notes
    -----
    Xavier uniform distribution: U(-limit, limit)
    where limit = sqrt(6 / (d_in + d_out)).
    """
    limit = np.sqrt(6.0 / (d_in + d_out))
    return rng.uniform(-limit, limit, size=(d_in, d_out))
