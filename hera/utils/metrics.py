import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon

def js_divergence(p_logits, q_logits):
    """Calculate Jensen-Shannon divergence between two distributions.
    
    Args:
        p_logits: Logits for distribution P
        q_logits: Logits for distribution Q
        
    Returns:
        JS divergence (scalar float)
    """
    # Convert to probabilities
    p = F.softmax(p_logits, dim=-1).cpu().detach().numpy()
    q = F.softmax(q_logits, dim=-1).cpu().detach().numpy()
    
    # Ensure numerical stability
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    
    # Handle multi-dimensional arrays
    if len(p.shape) > 1:
        # Average over batch and sequence dimensions
        p_flat = p.reshape(-1, p.shape[-1])
        q_flat = q.reshape(-1, q.shape[-1])
        js_values = np.array([jensenshannon(p_flat[i], q_flat[i]) 
                             for i in range(len(p_flat))])
        return float(js_values.mean())
    else:
        # Single distribution
        return float(jensenshannon(p, q))

def cosine_drift(a, b):
    """Calculate cosine drift between two activation tensors.
    
    Args:
        a: First activation tensor
        b: Second activation tensor
        
    Returns:
        Cosine drift (1 - cosine similarity) as scalar float
    """
    # Flatten if needed
    if a.dim() > 2:
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
    
    # Calculate cosine similarity and convert to drift
    sim = F.cosine_similarity(a, b, dim=-1).mean().item()
    return 1.0 - sim
