import sys
import os
import numpy as np

# Add src to sys.path to handle internal imports like 'model', 'utils', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend

def debug_keys():
    configuration = {
        "vocab_size": 32,
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_experts": 4,
        "max_seq_len": 32,
    }
    
    numpy_backend = NumPyBackend(**configuration)
    pytorch_backend = PyTorchBackend(**configuration)
    
    numpy_params = numpy_backend.get_params()
    pytorch_params = pytorch_backend.get_params()
    
    numpy_keys = set(numpy_params.keys())
    pytorch_keys = set(pytorch_params.keys())
    
    print("--- NumPy Keys (Truncated) ---")
    for k in sorted(list(numpy_keys))[:20]:
        print(k)
    print("...")
        
    print("\n--- PyTorch Keys (Truncated) ---")
    for k in sorted(list(pytorch_keys))[:20]:
        print(k)
    print("...")
        
    print("\n--- Key Mismatches ---")
    only_numpy = numpy_keys - pytorch_keys
    only_pytorch = pytorch_keys - numpy_keys
    
    if only_numpy:
        print(f"Only in NumPy (count: {len(only_numpy)}):")
        for k in sorted(list(only_numpy))[:20]:
            print(f"  {k}")
        if len(only_numpy) > 20:
            print("  ...")
    else:
        print("No keys only in NumPy")
        
    if only_pytorch:
        print(f"Only in PyTorch (count: {len(only_pytorch)}):")
        for k in sorted(list(only_pytorch))[:20]:
            print(f"  {k}")
        if len(only_pytorch) > 20:
            print("  ...")
    else:
        print("No keys only in PyTorch")

if __name__ == "__main__":
    debug_keys()
