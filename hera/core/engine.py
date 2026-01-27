import yaml
import torch
from transformer_lens import HookedTransformer
from hera.modules.sae_interface import SAEInterface
from hera.modules.evolutionary_layer import NeuroEvolutionaryLayer
from hera.modules.immune_system import ImmuneSystem
from hera.core.patch_registry import PatchRegistry
from hera.utils.logger import HeraLogger

class HeraEngine:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        self.logger = HeraLogger(self.cfg)
        self.logger.step("Initializing H.E.R.A.-R Engine...")

        self.model = HookedTransformer.from_pretrained(
            self.cfg["model"]["name"],
            device=self.cfg["experiment"]["device"]
        )
        self.logger.success(f"Model loaded: {self.cfg['model']['name']}")

        self.sae_if = SAEInterface(self.cfg)
        target_layer = self.cfg["evolution"]["target_layers"][0]
        self.evo_layer = NeuroEvolutionaryLayer(self.model, target_layer, self.sae_if, self.cfg)
        self.immune = ImmuneSystem(self.cfg, self.model)
        self.registry = PatchRegistry(self.cfg["logging"]["save_dir"])
        self.logger.success("System Ready.")

    def _measure_state(self, tokens):
        cache = {}
        def hook_fn(act, hook):
            cache["act"] = act.detach().clone()
        
        hook_point = f"blocks.{self.evo_layer.layer_idx}.hook_resid_pre"
        with self.model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
            logits = self.model(tokens)
            loss = self.model.loss(logits, tokens, per_token=True).mean()

        return {"logits": logits, "ppl": torch.exp(loss).item(), "activations": cache["act"]}

    def evolve(self, text):
        tokens = self.model.to_tokens(text)
        base = self._measure_state(tokens)
        delta_w = self.evo_layer.propose(base["activations"])
        
        if delta_w is None:
            return False

        self.evo_layer.commit(delta_w)
        mutated = self._measure_state(tokens)
        
        is_safe, reason, metrics = self.immune.verify(base, mutated, text)
        
        if is_safe:
            self.logger.success(f"Evolution Committed | JS: {metrics['js_div']:.4f}")
            self.registry.log_patch(self.evo_layer.layer_idx, metrics, text)
            return True
        else:
            self.evo_layer.rollback(delta_w)
            self.logger.warning(f"Rejected: {reason}")
            return False
            
    def shutdown(self):
        self.registry.save_history()