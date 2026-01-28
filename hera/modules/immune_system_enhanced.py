"""Enhanced ImmuneSystem with type hints, error handling, and additional safety checks."""

import torch
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from hera.utils.metrics import js_divergence, cosine_drift


class SafetyLevel(Enum):
    """Safety levels for evolution verification."""
    SAFE = "safe"  # All checks pass
    WARNING = "warning"  # Minor issues, may proceed with caution
    DANGEROUS = "dangerous"  # Significant issues, should reject
    CRITICAL = "critical"  # Critical issues, must reject


@dataclass
class SafetyMetrics:
    """Comprehensive safety metrics for evolution verification."""
    js_divergence: float  # Jensen-Shannon divergence
    ppl_ratio: float  # Perplexity ratio (mutated/baseline)
    activation_drift: float  # Cosine drift in activations
    entropy_change: float  # Change in output entropy
    top_k_agreement: float  # Agreement in top-k predictions
    safety_level: SafetyLevel
    rejection_reason: Optional[str] = None
    warning_messages: List[str] = None
    
    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []


class ImmuneSystem:
    """Enhanced digital immune system for online evolution safety.
    
    Monitors and verifies the safety of neuro-evolutionary updates
    using multiple metrics and reference probes.
    
    Args:
        cfg: Configuration dictionary
        model: Transformer model instance
    """
    
    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        """Initialize immune system with configuration and model."""
        self.cfg = cfg
        self.model = model
        
        # Validate configuration
        self._validate_config()
        
        # Load reference probes
        self.probes: List[Dict[str, Any]] = self._load_probes()
        
        # Initialize safety thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Track verification history
        self.history: List[SafetyMetrics] = []
        self.total_checks = 0
        self.rejections = 0
        
    def _validate_config(self) -> None:
        """Validate immune system configuration."""
        if "immune" not in self.cfg:
            raise KeyError("Missing 'immune' section in configuration")
        
        immune_cfg = self.cfg["immune"]
        required_keys = [
            "max_js_divergence",
            "max_ppl_spike", 
            "max_cosine_drift"
        ]
        
        for key in required_keys:
            if key not in immune_cfg:
                raise KeyError(f"Missing required immune config key: {key}")
    
    def _load_probes(self) -> List[Dict[str, Any]]:
        """Load reference probes for safety verification.
        
        Returns:
            List of probe dictionaries or empty list if file not found
        """
        probe_path = Path("data/reference_probes.json")
        
        if not probe_path.exists():
            print(f"Warning: Reference probes file not found at {probe_path}")
            return []
        
        try:
            with open(probe_path, 'r', encoding='utf-8') as f:
                probes = json.load(f)
            
            if not isinstance(probes, list):
                raise ValueError("Probes file should contain a list")
            
            print(f"Loaded {len(probes)} reference probes")
            return probes
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading probes: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error loading probes: {e}")
            return []
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize safety thresholds from config."""
        immune_cfg = self.cfg["immune"]
        
        return {
            "js_divergence": immune_cfg["max_js_divergence"],
            "ppl_ratio": immune_cfg["max_ppl_spike"],
            "cosine_drift": immune_cfg["max_cosine_drift"],
            "entropy_change": immune_cfg.get("max_entropy_change", 0.5),
            "top_k_disagreement": immune_cfg.get("max_top_k_disagreement", 0.3),
        }
    
    def verify(
        self,
        baseline: Dict[str, Any],
        mutated: Dict[str, Any],
        input_text: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Verify safety of evolution update.
        
        Args:
            baseline: Baseline model state
            mutated: Mutated model state
            input_text: Input text used for evolution
            
        Returns:
            Tuple of (is_safe, reason, metrics)
            
        Raises:
            ValueError: If input states are invalid
            RuntimeError: If verification fails
        """
        self.total_checks += 1
        
        try:
            # Validate inputs
            self._validate_states(baseline, mutated)
            
            # Calculate comprehensive safety metrics
            safety_metrics = self._calculate_safety_metrics(baseline, mutated)
            
            # Determine safety level
            is_safe, reason = self._evaluate_safety(safety_metrics)
            
            # Update history
            self.history.append(safety_metrics)
            if not is_safe:
                self.rejections += 1
            
            # Convert to legacy format for compatibility
            legacy_metrics = {
                "js_div": safety_metrics.js_divergence,
                "ppl_ratio": safety_metrics.ppl_ratio,
                "drift": safety_metrics.activation_drift,
                "entropy_change": safety_metrics.entropy_change,
                "top_k_agreement": safety_metrics.top_k_agreement,
                "safety_level": safety_metrics.safety_level.value,
            }
            
            return is_safe, reason, legacy_metrics
            
        except Exception as e:
            raise RuntimeError(f"Safety verification failed: {e}")
    
    def _validate_states(
        self,
        baseline: Dict[str, Any],
        mutated: Dict[str, Any]
    ) -> None:
        """Validate input states for verification."""
        required_keys = ["logits", "ppl", "activations"]
        
        for state in [baseline, mutated]:
            for key in required_keys:
                if key not in state:
                    raise ValueError(f"Missing required key '{key}' in state")
            
            if not isinstance(state["logits"], torch.Tensor):
                raise ValueError("Logits must be torch.Tensor")
            if not isinstance(state["activations"], torch.Tensor):
                raise ValueError("Activations must be torch.Tensor")
            if not isinstance(state["ppl"], (int, float)):
                raise ValueError("PPL must be numeric")
    
    def _calculate_safety_metrics(
        self,
        baseline: Dict[str, Any],
        mutated: Dict[str, Any]
    ) -> SafetyMetrics:
        """Calculate comprehensive safety metrics."""
        # Basic metrics
        js_div = js_divergence(baseline["logits"], mutated["logits"])
        ppl_ratio = mutated["ppl"] / (baseline["ppl"] + 1e-6)
        drift = cosine_drift(baseline["activations"], mutated["activations"])
        
        # Enhanced metrics
        entropy_change = self._calculate_entropy_change(
            baseline["logits"], mutated["logits"]
        )
        top_k_agreement = self._calculate_top_k_agreement(
            baseline["logits"], mutated["logits"]
        )
        
        # Initialize with basic metrics
        metrics = SafetyMetrics(
            js_divergence=float(js_div),
            ppl_ratio=float(ppl_ratio),
            activation_drift=float(drift),
            entropy_change=float(entropy_change),
            top_k_agreement=float(top_k_agreement),
            safety_level=SafetyLevel.SAFE,  # Will be updated
        )
        
        return metrics
    
    def _calculate_entropy_change(
        self,
        baseline_logits: torch.Tensor,
        mutated_logits: torch.Tensor
    ) -> float:
        """Calculate change in output entropy."""
        try:
            # Convert logits to probabilities
            baseline_probs = torch.softmax(baseline_logits, dim=-1)
            mutated_probs = torch.softmax(mutated_logits, dim=-1)
            
            # Calculate entropy
            baseline_entropy = -torch.sum(
                baseline_probs * torch.log(baseline_probs + 1e-10)
            ).item()
            mutated_entropy = -torch.sum(
                mutated_probs * torch.log(mutated_probs + 1e-10)
            ).item()
            
            # Calculate relative change
            if baseline_entropy > 0:
                return abs(mutated_entropy - baseline_entropy) / baseline_entropy
            else:
                return abs(mutated_entropy - baseline_entropy)
                
        except Exception:
            return 0.0  # Return safe default on error
    
    def _calculate_top_k_agreement(
        self,
        baseline_logits: torch.Tensor,
        mutated_logits: torch.Tensor,
        k: int = 5
    ) -> float:
        """Calculate agreement in top-k predictions."""
        try:
            # Get top-k indices
            _, baseline_topk = torch.topk(baseline_logits, k=k, dim=-1)
            _, mutated_topk = torch.topk(mutated_logits, k=k, dim=-1)
            
            # Calculate agreement
            agreement = 0.0
            total = baseline_topk.numel()
            
            for i in range(total):
                if baseline_topk.view(-1)[i] in mutated_topk.view(-1):
                    agreement += 1
            
            return agreement / total
            
        except Exception:
            return 1.0  # Return safe default on error
    
    def _evaluate_safety(
        self,
        metrics: SafetyMetrics
    ) -> Tuple[bool, str]:
        """Evaluate safety based on metrics."""
        warning_messages = []
        
        # Check each metric against thresholds
        if metrics.js_divergence > self.thresholds["js_divergence"]:
            metrics.safety_level = SafetyLevel.DANGEROUS
            return False, f"Output Divergence ({metrics.js_divergence:.4f})"
        
        if metrics.ppl_ratio > self.thresholds["ppl_ratio"]:
            metrics.safety_level = SafetyLevel.DANGEROUS
            return False, f"PPL Spike ({metrics.ppl_ratio:.2f}x)"
        
        if metrics.activation_drift > self.thresholds["cosine_drift"]:
            metrics.safety_level = SafetyLevel.DANGEROUS
            return False, f"Internal Drift ({metrics.activation_drift:.4f})"
        
        # Warning-level checks
        if metrics.entropy_change > self.thresholds["entropy_change"] * 0.7:
            warning_messages.append(
                f"High entropy change: {metrics.entropy_change:.3f}"
            )
            metrics.safety_level = SafetyLevel.WARNING
        
        if metrics.top_k_agreement < 1.0 - self.thresholds["top_k_disagreement"]:
            warning_messages.append(
                f"Top-k disagreement: {1.0 - metrics.top_k_agreement:.3f}"
            )
            metrics.safety_level = max(
                metrics.safety_level, SafetyLevel.WARNING
            )
        
        # Update warning messages
        metrics.warning_messages = warning_messages
        
        if metrics.safety_level == SafetyLevel.WARNING:
            return True, f"Stable (warnings: {', '.join(warning_messages)})"
        else:
            return True, "Stable"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get immune system statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        if self.total_checks == 0:
            return {
                "total_checks": 0,
                "rejection_rate": 0.0,
                "avg_js_div": 0.0,
                "avg_ppl_ratio": 0.0,
            }
        
        rejection_rate = self.rejections / self.total_checks
        
        # Calculate average metrics from history
        if self.history:
            avg_js_div = np.mean([m.js_divergence for m in self.history])
            avg_ppl_ratio = np.mean([m.ppl_ratio for m in self.history])
            avg_drift = np.mean([m.activation_drift for m in self.history])
        else:
            avg_js_div = avg_ppl_ratio = avg_drift = 0.0
        
        return {
            "total_checks": self.total_checks,
            "rejections": self.rejections,
            "rejection_rate": rejection_rate,
            "avg_js_divergence": float(avg_js_div),
            "avg_ppl_ratio": float(avg_ppl_ratio),
            "avg_activation_drift": float(avg_drift),
            "probe_count": len(self.probes),
        }
    
    def add_probe(self, text: str, expected_output: str) -> None:
        """Add a new reference probe.
        
        Args:
            text: Input text for the probe
            expected_output: Expected model output
        """
        probe = {
            "text": text,
            "expected_output": expected_output,
            "timestamp": np.datetime64('now').astype(str),
        }
        
        self.probes.append(probe)
        
        # Save probes to file
        self._save_probes()
    
    def _save_probes(self) -> None:
        """Save probes to file."""
        probe_path = Path("data/reference_probes.json")
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(probe_path, 'w', encoding='utf-8') as f:
                json.dump(self.probes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save probes: {e}")
    
    def run_probe_tests(self) -> Dict[str, Any]:
        """Run all reference probes and return results.
        
        Returns:
            Dictionary with probe test results
        """
        if not self.probes:
            return {"total_probes": 0, "passed": 0, "failed": 0}
        
        results = {
            "total_probes": len(self.probes),
            "passed": 0,
            "failed": 0,
            "details": [],
        }
        
        for i, probe in enumerate(self.probes):
            try:
                # Tokenize probe text
                tokens = self.model.to_tokens(probe["text"])
                
                # Run model
                with torch.no_grad():
                    logits = self.model(tokens)
                    loss = self.model.loss(logits, tokens, per_token=True).mean()
                    ppl = torch.exp(loss).item()
                
                # Simple check: model should generate reasonable output
                # (In practice, you'd want more sophisticated checks)
                is_ok = ppl < 100.0  # Arbitrary threshold
                
                if is_ok:
                    results["passed"] += 1
                    status = "PASS"
                else:
                    results["failed"] += 1
                    status = "FAIL"
                
                results["details"].append({
                    "probe_id": i,
                    "text": probe["text"][:50] + "...",
                    "ppl": ppl,
                    "status": status,
                })
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "probe_id": i,
                    "text": probe["text"][:50] + "...",
                    "error": str(e),
                    "status": "ERROR",
                })
        
        return results