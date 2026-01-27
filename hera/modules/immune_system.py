import torch
import json
from hera.utils.metrics import js_divergence, cosine_drift

class ImmuneSystem:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        try:
            with open("data/reference_probes.json", "r") as f:
                self.probes = json.load(f)
        except:
            self.probes = []

    def verify(self, baseline, mutated, input_text):
        js = js_divergence(baseline["logits"], mutated["logits"])
        ppl_ratio = mutated["ppl"] / (baseline["ppl"] + 1e-6)
        drift = cosine_drift(baseline["activations"], mutated["activations"])

        metrics = {"js_div": float(js), "ppl_ratio": float(ppl_ratio), "drift": float(drift)}

        if js > self.cfg["immune"]["max_js_divergence"]:
            return False, f"Output Divergence ({js:.4f})", metrics
        if ppl_ratio > self.cfg["immune"]["max_ppl_spike"]:
            return False, f"PPL Spike ({ppl_ratio:.2f}x)", metrics
        if drift > self.cfg["immune"]["max_cosine_drift"]:
            return False, f"Internal Drift ({drift:.4f})", metrics

        return True, "Stable", metrics