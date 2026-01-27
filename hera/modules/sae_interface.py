import torch
from sae_lens import SAE

class SAEInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sae, _, _ = SAE.from_pretrained(
            release=cfg["sae"]["release"],
            sae_id=cfg["sae"]["id"],
            device=cfg["experiment"]["device"]
        )
        self.sae.eval()

    def encode(self, activations):
        with torch.no_grad():
            return self.sae.encode(activations)

    def decode_direction(self, feature_idx):
        return self.sae.W_dec[feature_idx].detach()