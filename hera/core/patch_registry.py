import time
import json
import os

class PatchRegistry:
    def __init__(self, save_dir):
        self.history = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log_patch(self, layer, metrics, prompt_snippet):
        entry = {
            "timestamp": time.time(),
            "layer": layer,
            "prompt": prompt_snippet[:50] + "...",
            "metrics": metrics,
            "status": "COMMITTED"
        }
        self.history.append(entry)
    
    def save_history(self):
        with open(f"{self.save_dir}/evolution_history.json", "w") as f:
            json.dump(self.history, f, indent=2)