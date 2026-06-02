import numpy as np
from src.backends.numpy.numpy_backend import NumPyBackend
from src.backends.pytorch.pytorch_backend import PyTorchBackend

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
    
    print("--- NumPy Keys ---")
    for k in sorted(numpy_keys):
        print(k)
        
    print("\n--- PyTorch Keys ---")
    for k in sorted(pytorch_keys):
        print(k)
        
    print("\n--- Key Mismatches ---")
    only_numpy = numpy_keys - pytorch_keys
    only_pytorch = pytorch_keys - numpy_keys
    
    if only_numpy:
        print(f"Only in NumPy: {only_numpy}")
    else:
        print("No keys only in NumPy")
        
    if only_pytorch:
        print(f"Only in PyTorch: {only_pytorch}")
    else:
        print("No keys only in PyTorch")

if __name__ == "__main__":
    debug_keys()
