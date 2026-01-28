"""Enhanced NeuroEvolutionaryLayer with type hints, error handling, and additional features."""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class EvolutionStrategy(Enum):
    """Evolution strategies for weight updates."""
    HEBBIAN = "hebbian"  # Outer product of feature directions
    GRADIENT = "gradient"  # Gradient-based updates
    RANDOM = "random"  # Random exploration
    MIXED = "mixed"  # Combination of strategies


@dataclass
class EvolutionMetrics:
    """Metrics for tracking evolution performance."""
    feature_count: int
    max_activation: float
    mean_activation: float
    weight_change_norm: float
    strategy_used: EvolutionStrategy
    success: bool


class NeuroEvolutionaryLayer:
    """Enhanced layer for neuro-evolutionary weight updates.
    
    This class implements Hebbian-like weight updates based on SAE features,
    with additional strategies and safety features.
    
    Args:
        model: Transformer model instance
        layer_idx: Index of layer to evolve
        sae_interface: SAE interface for feature extraction
        cfg: Configuration dictionary
    """
    
    def __init__(
        self,
        model: Any,
        layer_idx: int,
        sae_interface: Any,
        cfg: Dict[str, Any]
    ) -> None:
        """Initialize neuro-evolutionary layer."""
        self.model = model
        self.layer_idx = layer_idx
        self.sae = sae_interface
        self.cfg = cfg
        
        # Validate configuration
        self._validate_config()
        
        # Initialize target module
        self.target_module = self._get_target_module()
        self.initial_weight = self.target_module.W.data.clone()
        
        # Track evolution history
        self.history: List[EvolutionMetrics] = []
        self.total_updates = 0
        self.successful_updates = 0
        
        # Initialize evolution strategy
        self.strategy = EvolutionStrategy(
            self.cfg.get("evolution", {}).get("strategy", "hebbian")
        )
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ["evolution", "sae"]
        for key in required_keys:
            if key not in self.cfg:
                raise KeyError(f"Missing required config key: {key}")
        
        evolution_cfg = self.cfg["evolution"]
        required_evolution_keys = ["learning_rate", "top_k_features"]
        for key in required_evolution_keys:
            if key not in evolution_cfg:
                raise KeyError(f"Missing required evolution config key: {key}")
        
        sae_cfg = self.cfg["sae"]
        if "threshold" not in sae_cfg:
            raise KeyError("Missing SAE threshold in config")
    
    def _get_target_module(self) -> Any:
        """Get target module for weight updates."""
        try:
            if self.layer_idx >= len(self.model.blocks):
                raise ValueError(
                    f"Layer index {self.layer_idx} out of bounds "
                    f"(model has {len(self.model.blocks)} layers)"
                )
            
            block = self.model.blocks[self.layer_idx]
            if not hasattr(block, 'mlp'):
                raise AttributeError(f"Block {self.layer_idx} has no MLP attribute")
            
            if not hasattr(block.mlp, 'W_out'):
                raise AttributeError(f"MLP has no W_out attribute")
            
            return block.mlp.W_out
            
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Failed to get target module: {e}")
    
    def propose(
        self,
        activations: torch.Tensor,
        strategy: Optional[EvolutionStrategy] = None
    ) -> Optional[torch.Tensor]:
        """Propose weight update based on activations.
        
        Args:
            activations: Input activations tensor
            strategy: Evolution strategy to use (defaults to configured strategy)
            
        Returns:
            Weight update tensor or None if no valid features
            
        Raises:
            ValueError: If activations have wrong shape
            RuntimeError: If feature extraction fails
        """
        if activations.dim() != 3:
            raise ValueError(
                f"Activations must be 3D tensor, got shape {activations.shape}"
            )
        
        # Use specified strategy or default
        current_strategy = strategy or self.strategy
        
        try:
            # Extract features using SAE
            feature_acts = self.sae.encode(activations)
            
            # Get last token activations (most relevant for generation)
            last_token_acts = feature_acts[0, -1, :]
            
            # Select top-k features
            top_k = self.cfg["evolution"]["top_k_features"]
            top_vals, top_indices = torch.topk(last_token_acts, k=top_k)
            
            # Filter by threshold
            threshold = self.cfg["sae"]["threshold"]
            mask = top_vals > threshold
            valid_indices = top_indices[mask]
            
            if len(valid_indices) == 0:
                # No features above threshold
                metrics = EvolutionMetrics(
                    feature_count=0,
                    max_activation=float(top_vals.max().item()),
                    mean_activation=float(top_vals.mean().item()),
                    weight_change_norm=0.0,
                    strategy_used=current_strategy,
                    success=False
                )
                self.history.append(metrics)
                return None
            
            # Calculate weight update based on strategy
            if current_strategy == EvolutionStrategy.HEBBIAN:
                delta_w = self._hebbian_update(valid_indices)
            elif current_strategy == EvolutionStrategy.GRADIENT:
                delta_w = self._gradient_update(activations, valid_indices)
            elif current_strategy == EvolutionStrategy.RANDOM:
                delta_w = self._random_update(valid_indices)
            elif current_strategy == EvolutionStrategy.MIXED:
                delta_w = self._mixed_update(activations, valid_indices)
            else:
                raise ValueError(f"Unknown evolution strategy: {current_strategy}")
            
            # Apply learning rate
            learning_rate = self.cfg["evolution"]["learning_rate"]
            delta_w = delta_w * learning_rate
            
            # Track metrics
            weight_norm = delta_w.norm().item()
            metrics = EvolutionMetrics(
                feature_count=len(valid_indices),
                max_activation=float(top_vals.max().item()),
                mean_activation=float(top_vals.mean().item()),
                weight_change_norm=weight_norm,
                strategy_used=current_strategy,
                success=True
            )
            self.history.append(metrics)
            
            return delta_w
            
        except Exception as e:
            raise RuntimeError(f"Failed to propose weight update: {e}")
    
    def _hebbian_update(self, feature_indices: torch.Tensor) -> torch.Tensor:
        """Hebbian-like update: ΔW = Σ dᵢdᵢᵀ."""
        delta_w = torch.zeros_like(self.target_module.W.data)
        
        for idx in feature_indices:
            direction = self.sae.decode_direction(idx.item())
            update = torch.outer(direction, direction)
            delta_w += update
        
        # Normalize by number of features
        if len(feature_indices) > 0:
            delta_w = delta_w / len(feature_indices)
        
        return delta_w
    
    def _gradient_update(
        self,
        activations: torch.Tensor,
        feature_indices: torch.Tensor
    ) -> torch.Tensor:
        """Gradient-based update (placeholder for future implementation)."""
        # For now, fall back to Hebbian
        return self._hebbian_update(feature_indices)
    
    def _random_update(self, feature_indices: torch.Tensor) -> torch.Tensor:
        """Random exploration update."""
        shape = self.target_module.W.data.shape
        return torch.randn(shape, device=self.target_module.W.data.device)
    
    def _mixed_update(
        self,
        activations: torch.Tensor,
        feature_indices: torch.Tensor
    ) -> torch.Tensor:
        """Mixed strategy update."""
        hebbian = self._hebbian_update(feature_indices)
        random = self._random_update(feature_indices)
        
        # 70% Hebbian, 30% random
        return 0.7 * hebbian + 0.3 * random
    
    def commit(self, delta_w: torch.Tensor) -> None:
        """Commit weight update.
        
        Args:
            delta_w: Weight update tensor
            
        Raises:
            ValueError: If delta_w has wrong shape
            RuntimeError: If update fails
        """
        if delta_w.shape != self.target_module.W.data.shape:
            raise ValueError(
                f"Delta shape {delta_w.shape} doesn't match "
                f"weight shape {self.target_module.W.data.shape}"
            )
        
        try:
            self.target_module.W.data += delta_w
            self.total_updates += 1
            self.successful_updates += 1
            
        except Exception as e:
            raise RuntimeError(f"Failed to commit weight update: {e}")
    
    def rollback(self, delta_w: torch.Tensor) -> None:
        """Rollback weight update.
        
        Args:
            delta_w: Weight update tensor to rollback
            
        Raises:
            ValueError: If delta_w has wrong shape
            RuntimeError: If rollback fails
        """
        if delta_w.shape != self.target_module.W.data.shape:
            raise ValueError(
                f"Delta shape {delta_w.shape} doesn't match "
                f"weight shape {self.target_module.W.data.shape}"
            )
        
        try:
            self.target_module.W.data -= delta_w
            
        except Exception as e:
            raise RuntimeError(f"Failed to rollback weight update: {e}")
    
    def reset(self) -> None:
        """Reset weights to initial state."""
        try:
            self.target_module.W.data.copy_(self.initial_weight)
            self.history.clear()
            self.total_updates = 0
            self.successful_updates = 0
            
        except Exception as e:
            raise RuntimeError(f"Failed to reset weights: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evolution metrics.
        
        Returns:
            Dictionary with evolution statistics
        """
        if not self.history:
            return {
                "total_updates": self.total_updates,
                "successful_updates": self.successful_updates,
                "success_rate": 0.0,
                "recent_history": [],
            }
        
        recent_history = [
            {
                "feature_count": m.feature_count,
                "max_activation": m.max_activation,
                "weight_change_norm": m.weight_change_norm,
                "strategy": m.strategy_used.value,
                "success": m.success,
            }
            for m in self.history[-10:]  # Last 10 updates
        ]
        
        success_rate = (
            self.successful_updates / self.total_updates
            if self.total_updates > 0 else 0.0
        )
        
        return {
            "total_updates": self.total_updates,
            "successful_updates": self.successful_updates,
            "success_rate": success_rate,
            "current_strategy": self.strategy.value,
            "recent_history": recent_history,
        }
    
    def set_strategy(self, strategy: EvolutionStrategy) -> None:
        """Set evolution strategy.
        
        Args:
            strategy: New evolution strategy
        """
        self.strategy = strategy
    
    def get_weight_change(self) -> torch.Tensor:
        """Get total weight change from initial state.
        
        Returns:
            Weight difference tensor
        """
        return self.target_module.W.data - self.initial_weight
    
    def get_feature_contributions(
        self,
        activations: torch.Tensor
    ) -> Dict[int, float]:
        """Get contribution of each feature to weight updates.
        
        Args:
            activations: Input activations
            
        Returns:
            Dictionary mapping feature indices to contribution scores
        """
        try:
            feature_acts = self.sae.encode(activations)
            last_token_acts = feature_acts[0, -1, :]
            
            # Get top features
            top_k = self.cfg["evolution"]["top_k_features"]
            top_vals, top_indices = torch.topk(last_token_acts, k=top_k)
            
            contributions = {}
            for val, idx in zip(top_vals, top_indices):
                if val > self.cfg["sae"]["threshold"]:
                    contributions[idx.item()] = val.item()
            
            return contributions
            
        except Exception as e:
            raise RuntimeError(f"Failed to get feature contributions: {e}")