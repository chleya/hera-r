import yaml
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from transformer_lens import HookedTransformer
from hera.modules.sae_interface import SAEInterface
from hera.modules.evolutionary_layer_enhanced import NeuroEvolutionaryLayer, EvolutionStrategy
from hera.modules.immune_system_enhanced import ImmuneSystem
from hera.core.patch_registry import PatchRegistry
from hera.utils.logger import HeraLogger


class HeraEngine:
    """Main orchestrator for H.E.R.A.-R online evolution framework.
    
    This class coordinates all components of the online evolution system:
    - Model loading and inference
    - Sparse Autoencoder (SAE) feature extraction
    - Neuro-evolutionary weight updates
    - Digital immune system safety monitoring
    - Patch registry for tracking changes
    
    Args:
        config_path: Path to YAML configuration file
    """
    
    def __init__(self, config_path: str) -> None:
        """Initialize HeraEngine with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            KeyError: If required config sections are missing
            RuntimeError: If model or SAE loading fails
        """
        # Validate config path
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.cfg: Dict[str, Any] = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
        
        # Validate required config sections
        required_sections = ["experiment", "model", "sae", "evolution", "immune", "logging"]
        for section in required_sections:
            if section not in self.cfg:
                raise KeyError(f"Missing required config section: {section}")
        
        # Initialize logger
        self.logger = HeraLogger(self.cfg)
        self.logger.step("Initializing H.E.R.A.-R Engine...")
        
        try:
            # Load model
            model_name = self.cfg["model"]["name"]
            device = self.cfg["experiment"]["device"]
            
            self.logger.info(f"Loading model: {model_name} on {device}")
            self.model: HookedTransformer = HookedTransformer.from_pretrained(
                model_name,
                device=device
            )
            self.model.eval()  # Set to evaluation mode
            self.logger.success(f"Model loaded: {model_name}")
            
            # Initialize SAE interface
            self.sae_if: SAEInterface = SAEInterface(self.cfg)
            
            # Initialize evolutionary layer
            target_layer = self.cfg["evolution"]["target_layers"][0]
            self.evo_layer: NeuroEvolutionaryLayer = NeuroEvolutionaryLayer(
                self.model, target_layer, self.sae_if, self.cfg
            )
            
            # Initialize immune system
            self.immune: ImmuneSystem = ImmuneSystem(self.cfg, self.model)
            
            # Initialize patch registry
            save_dir = self.cfg["logging"]["save_dir"]
            self.registry: PatchRegistry = PatchRegistry(save_dir)
            
            self.logger.success("H.E.R.A.-R Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HeraEngine: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def _measure_state(self, tokens: torch.Tensor) -> Dict[str, Any]:
        """Measure model state for given tokens.
        
        Args:
            tokens: Input tokens tensor of shape (batch, seq_len)
            
        Returns:
            Dictionary containing:
            - logits: Model output logits
            - ppl: Perplexity of the input
            - activations: Activations at target layer
            
        Raises:
            RuntimeError: If hook capture fails
        """
        cache: Dict[str, torch.Tensor] = {}
        
        def hook_fn(act: torch.Tensor, hook) -> None:
            """Hook function to capture activations."""
            cache["act"] = act.detach().clone()
        
        hook_point = f"blocks.{self.evo_layer.layer_idx}.hook_resid_pre"
        
        try:
            with self.model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
                logits = self.model(tokens)
                loss = self.model.loss(logits, tokens, per_token=True).mean()
            
            if "act" not in cache:
                raise RuntimeError("Failed to capture activations from hook")
            
            return {
                "logits": logits,
                "ppl": torch.exp(loss).item(),
                "activations": cache["act"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to measure state: {e}")
            raise RuntimeError(f"State measurement failed: {e}")

    def evolve(self, text: str) -> bool:
        """Perform online evolution with given text input.
        
        Args:
            text: Input text prompt for evolution
            
        Returns:
            True if evolution was committed, False if rejected
            
        Raises:
            ValueError: If input text is empty
            RuntimeError: If evolution process fails
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        self.logger.info(f"Evolving with prompt: {text[:50]}...")
        
        try:
            # Tokenize input
            tokens = self.model.to_tokens(text)
            if tokens.numel() == 0:
                raise ValueError("Tokenization produced empty tensor")
            
            # Measure baseline state
            base_state = self._measure_state(tokens)
            
            # Propose weight update
            delta_w = self.evo_layer.propose(base_state["activations"])
            if delta_w is None:
                self.logger.info("No evolution proposed (no features above threshold)")
                return False
            
            # Apply weight update
            self.evo_layer.commit(delta_w)
            
            # Measure mutated state
            mutated_state = self._measure_state(tokens)
            
            # Verify safety with immune system
            is_safe, reason, metrics = self.immune.verify(
                base_state, mutated_state, text
            )
            
            if is_safe:
                # Evolution accepted
                self.logger.success(
                    f"Evolution Committed | "
                    f"JS: {metrics.get('js_div', 0):.4f}, "
                    f"PPL: {metrics.get('ppl_ratio', 0):.2f}x, "
                    f"Drift: {metrics.get('drift', 0):.4f}"
                )
                
                # Log to registry
                self.registry.log_patch(
                    layer_idx=self.evo_layer.layer_idx,
                    metrics=metrics,
                    context=text
                )
                return True
            else:
                # Evolution rejected - rollback
                self.logger.warning(f"Evolution Rejected: {reason}")
                self.evo_layer.rollback(delta_w)
                return False
                
        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
            # Attempt to recover if we were in the middle of an update
            if 'delta_w' in locals() and delta_w is not None:
                try:
                    self.evo_layer.rollback(delta_w)
                    self.logger.info("Rolled back weight update after error")
                except Exception as rollback_error:
                    self.logger.error(f"Failed to rollback after error: {rollback_error}")
            raise RuntimeError(f"Evolution process failed: {e}")
            
    def shutdown(self) -> None:
        """Clean shutdown of HeraEngine.
        
        Saves registry history and performs cleanup.
        """
        self.logger.step("Shutting down H.E.R.A.-R Engine...")
        
        try:
            self.registry.save_history()
            self.logger.success("Registry history saved")
            
            # Additional cleanup if needed
            if hasattr(self, 'model'):
                # Clear CUDA cache if using GPU
                if self.cfg["experiment"]["device"] == "cuda":
                    torch.cuda.empty_cache()
            
            self.logger.success("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise RuntimeError(f"Shutdown failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of HeraEngine.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            "model": self.cfg["model"]["name"],
            "device": self.cfg["experiment"]["device"],
            "target_layer": self.evo_layer.layer_idx,
            "registry_entries": len(self.registry.history) if hasattr(self.registry, 'history') else 0,
            "config_hash": hash(str(self.cfg)),  # Simple config hash
        }
    
    def reset(self) -> None:
        """Reset engine to initial state.
        
        Clears registry and resets evolutionary layer weights.
        """
        self.logger.step("Resetting H.E.R.A.-R Engine...")
        
        try:
            # Reset evolutionary layer weights
            self.evo_layer.reset()
            
            # Clear registry
            self.registry.clear()
            
            self.logger.success("Engine reset complete")
            
        except Exception as e:
            self.logger.error(f"Error during reset: {e}")
            raise RuntimeError(f"Reset failed: {e}")