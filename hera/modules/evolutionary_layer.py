import torch

class NeuroEvolutionaryLayer:
    def __init__(self, model, layer_idx, sae_interface, cfg):
        self.model = model
        self.layer_idx = layer_idx
        self.sae = sae_interface
        self.cfg = cfg
        self.target_module = model.blocks[layer_idx].mlp.W_out
        self.initial_weight = self.target_module.W.data.clone()

    def propose(self, activations):
        feature_acts = self.sae.encode(activations)
        last_token_acts = feature_acts[0, -1, :] 
        top_vals, top_indices = torch.topk(last_token_acts, k=self.cfg["evolution"]["top_k_features"])
        
        mask = top_vals > self.cfg["sae"]["threshold"]
        valid_indices = top_indices[mask]
        
        if len(valid_indices) == 0:
            return None

        delta_w_total = torch.zeros_like(self.target_module.W.data)
        for idx in valid_indices:
            d = self.sae.decode_direction(idx)
            update = torch.outer(d, d)
            delta_w_total += update
            
        delta_w_total = delta_w_total / (len(valid_indices) + 1e-6)
        return delta_w_total * self.cfg["evolution"]["learning_rate"]

    def commit(self, delta_w):
        self.target_module.W.data += delta_w

    def rollback(self, delta_w):
        self.target_module.W.data -= delta_w