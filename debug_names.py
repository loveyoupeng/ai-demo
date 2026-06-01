import numpy as np
import torch
from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend

def check_names(configuration):
    numpy_backend = NumPyBackend(**configuration)
    pytorch_backend = PyTorchBackend(**configuration)

    print("NumPy Param Names:")
    for name in numpy_backend.get_params().keys():
        print(f"  {name}")

    print("\nPyTorch Param Names:")
    for name in pytorch_backend.get_params().keys():
        print(f"  {name}")

if __name__ == "__main__":
    configuration = {
        "vocab_size": 32,
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_experts": 4,
        "max_seq_len": 32,
    }
    check_names(configuration)
