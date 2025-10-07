import torch
import math

def sinusoidal_embedding(N, D):
    """
    Create sinusoidal positional embeddings for positions 1 to N
    Args:
        N: number of positions (assumes positions 1 to N)
        D: embedding dimension (must be even)
    Returns:
        Tensor of shape [N, D]
    """
    position = torch.arange(1, N + 1).unsqueeze(1)         # shape [N, 1]
    div_term = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))  # shape [D/2]
    
    pe = torch.zeros(N, D)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    
    return pe
