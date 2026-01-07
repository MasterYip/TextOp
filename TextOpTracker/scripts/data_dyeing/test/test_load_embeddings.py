"""
Minimal test script to load and verify saved motion embeddings.

Usage:
    python test_load_embeddings.py episode_3_embeddings_interval50.pt
    python test_load_embeddings.py episode_3_embeddings_interval50.txt
    python test_load_embeddings.py episode_3_embeddings_interval50.npy
"""

import sys
import torch
import numpy as np
from pathlib import Path


def load_embeddings(file_path):
    """Load embeddings from txt, npy, or pt file."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.pt':
        embeddings = torch.load(file_path)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
    elif file_path.suffix == '.npy':
        embeddings = np.load(file_path)
    elif file_path.suffix == '.txt':
        embeddings = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    return embeddings


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_load_embeddings.py <embedding_file>")
        print("Example: python test_load_embeddings.py episode_3_embeddings_interval50.pt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"Loading embeddings from: {file_path}")
    embeddings = load_embeddings(file_path)
    
    print(f"\n{'='*60}")
    print("EMBEDDING INFO")
    print(f"{'='*60}")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Min: {embeddings.min():.6f}")
    print(f"Max: {embeddings.max():.6f}")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    print(f"\nFirst embedding (first 10 values):")
    print(embeddings[0, :10])
    print(f"\nLast embedding (first 10 values):")
    print(embeddings[-1, :10])
    print(f"{'='*60}\n")
    
    # Test conversion to torch tensor for policy use
    cond_tensor = torch.from_numpy(embeddings).float()
    print(f"Converted to torch tensor: {cond_tensor.shape}")
    print(f"Ready for use as conditioning: cond = embeddings[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 512]")
    print(f"Example shape for batch=16: {cond_tensor[0].unsqueeze(0).expand(16, 1, -1).shape}")


if __name__ == '__main__':
    main()
    # python test_load_embeddings.py episode_3_embeddings_interval50.pt