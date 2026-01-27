import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon

def js_divergence(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1).cpu().detach().numpy()
    q = F.softmax(q_logits, dim=-1).cpu().detach().numpy()
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    if len(p.shape) > 1:
        return np.mean([jensenshannon(p[i], q[i]) for i in range(len(p))])
    return jensenshannon(p, q)

def cosine_drift(a, b):
    if a.dim() > 2:
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
    sim = F.cosine_similarity(a, b, dim=-1).mean().item()
    return 1 - sim