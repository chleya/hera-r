import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon

def js_divergence(p, q):
    p = p.cpu().numpy()
    q = q.cpu().numpy()
    return jensenshannon(p, q)

def cosine_drift(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1).mean().item()
