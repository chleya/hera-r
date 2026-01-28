#!/usr/bin/env python3
"""
Research Experiment Template for H.E.R.A.-R

This template provides a standardized structure for conducting research experiments
with the H.E.R.A.-R framework. Researchers can extend this template for their
specific experimental needs.
"""

import yaml
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from hera.core.engine import HeraEngine
from hera.utils.logger import HeraLogger


class ResearchExperiment:
    """Base class for H.E.R.A.-R research experiments."""
    
    def __init__(self, experiment_name: str, config_path: str = "configs/default.yaml"):
        """
        Initialize a research experiment.
        
        Args:
            experiment_name: Unique name for this experiment
            config_path: Path to configuration file
        """
        self.experiment_name = experiment_name
        self.config_path = config_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{experiment_name}_{self.timestamp}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Initialize logger
        self.logger = HeraLogger(self.base_config)
        self.logger.success(f"Initialized experiment: {experiment_name}")
        
        # Results storage
        self.results = {
            "experiment_info": {
                "name": experiment_name,
                "timestamp": self.timestamp,
                "config_path": config_path,
            },
            "config": self.base_config,
            "evolution_steps": [],
            "metrics": {},
            "analysis": {},
        }
    
    def setup_engine(self, custom_config: Optional[Dict] = None) -> HeraEngine:
        """
        Set up HeraEngine with optional custom configuration.
        
        Args:
            custom_config: Optional configuration overrides
            
        Returns:
            Initialized HeraEngine instance
        """
        config = self.base_config.copy()
        if custom_config:
            # Deep merge custom config
            self._deep_merge(config, custom_config)
        
        # Save experiment-specific config
        config_path = self.experiment_dir / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.step("Initializing HeraEngine...")
        engine = HeraEngine(str(config_path))
        
        # Save engine info
        self.results["engine_info"] = {
            "model": config.get("model", {}).get("name"),
            "device": config.get("experiment", {}).get("device"),
            "config_path": str(config_path),
        }
        
        return engine
    
    def run_evolution_sequence(self, engine: HeraEngine, prompts: List[str]) -> Dict[str, Any]:
        """
        Run a sequence of evolution steps.
        
        Args:
            engine: HeraEngine instance
            prompts: List of prompts for evolution
            
        Returns:
            Dictionary with evolution results
        """
        self.logger.step(f"Running evolution sequence with {len(prompts)} prompts")
        
        evolution_results = {
            "prompts": prompts,
            "steps": [],
            "summary": {},
        }
        
        successful_evolutions = 0
        rejected_evolutions = 0
        
        for i, prompt in enumerate(prompts):
            step_result = self._run_single_evolution(engine, prompt, step_idx=i)
            evolution_results["steps"].append(step_result)
            
            if step_result["success"]:
                successful_evolutions += 1
            else:
                rejected_evolutions += 1
            
            # Log progress
            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                self.logger.info(f"Progress: {i+1}/{len(prompts)} steps completed")
        
        # Calculate summary statistics
        evolution_results["summary"] = {
            "total_steps": len(prompts),
            "successful": successful_evolutions,
            "rejected": rejected_evolutions,
            "success_rate": successful_evolutions / len(prompts) if prompts else 0,
            "rejection_rate": rejected_evolutions / len(prompts) if prompts else 0,
        }
        
        self.logger.success(f"Evolution sequence completed: {successful_evolutions} successful, {rejected_evolutions} rejected")
        
        return evolution_results
    
    def _run_single_evolution(self, engine: HeraEngine, prompt: str, step_idx: int) -> Dict[str, Any]:
        """
        Run a single evolution step.
        
        Args:
            engine: HeraEngine instance
            prompt: Input prompt for evolution
            step_idx: Step index for tracking
            
        Returns:
            Dictionary with step results
        """
        step_result = {
            "step": step_idx,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "reason": "",
            "metrics": {},
            "registry_entry": None,
        }
        
        try:
            # Run evolution
            success = engine.evolve(prompt)
            
            # Get latest registry entry if available
            if hasattr(engine, 'registry') and engine.registry.history:
                step_result["registry_entry"] = engine.registry.history[-1]
                step_result["metrics"] = engine.registry.history[-1].get("metrics", {})
            
            step_result["success"] = success
            step_result["reason"] = "Committed" if success else "Rejected by immune system"
            
            # Log result
            status_emoji = "✅" if success else "🛑"
            self.logger.info(f"Step {step_idx}: {status_emoji} {prompt[:50]}...")
            
        except Exception as e:
            step_result["success"] = False
            step_result["reason"] = f"Error: {str(e)}"
            self.logger.error(f"Step {step_idx} failed: {str(e)}")
        
        return step_result
    
    def measure_stability(self, engine: HeraEngine, probe_sentences: List[str]) -> Dict[str, Any]:
        """
        Measure model stability using probe sentences.
        
        Args:
            engine: HeraEngine instance
            probe_sentences: Sentences to test stability
            
        Returns:
            Dictionary with stability metrics
        """
        self.logger.step("Measuring model stability...")
        
        stability_results = {
            "probe_sentences": probe_sentences,
            "measurements": [],
            "summary": {},
        }
        
        # This would typically involve:
        # 1. Generating outputs for each probe sentence
        # 2. Comparing before/after evolution
        # 3. Calculating various stability metrics
        
        # Placeholder implementation
        for i, sentence in enumerate(probe_sentences):
            measurement = {
                "sentence": sentence,
                "perplexity": np.random.uniform(5, 20),  # Placeholder
                "output_consistency": np.random.uniform(0.7, 1.0),  # Placeholder
            }
            stability_results["measurements"].append(measurement)
        
        # Calculate summary statistics
        perplexities = [m["perplexity"] for m in stability_results["measurements"]]
        consistencies = [m["output_consistency"] for m in stability_results["measurements"]]
        
        stability_results["summary"] = {
            "mean_perplexity": np.mean(perplexities),
            "std_perplexity": np.std(perplexities),
            "mean_consistency": np.mean(consistencies),
            "std_consistency": np.std(consistencies),
        }
        
        self.logger.success("Stability measurement completed")
        
        return stability_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """
        Save experiment results to file.
        
        Args:
            results: Results dictionary to save
            filename: Name of results file
        """
        results_path = self.experiment_dir / filename
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.success(f"Results saved to: {results_path}")
        
        # Also save a summary file
        summary = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "results_summary": self._extract_summary(serializable_results),
        }
        
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _extract_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key summary information from results."""
        summary = {}
        
        if "evolution_results" in results and "summary" in results["evolution_results"]:
            summary["evolution"] = results["evolution_results"]["summary"]
        
        if "stability_results" in results and "summary" in results["stability_results"]:
            summary["stability"] = results["stability_results"]["summary"]
        
        return summary
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable experiment report.
        
        Args:
            results: Experiment results
            
        Returns:
            Markdown formatted report
        """
        report = f"""# H.E.R.A.-R Research Experiment Report

## Experiment Information
- **Name**: {self.experiment_name}
- **Timestamp**: {self.timestamp}
- **Configuration**: {self.config_path}

## Summary

"""
        
        # Add evolution summary
        if "evolution_results" in results and "summary" in results["evolution_results"]:
            evo_summary = results["evolution_results"]["summary"]
            report += f"""### Evolution Results
- **Total Steps**: {evo_summary.get('total_steps', 0)}
- **Successful**: {evo_summary.get('successful', 0)}
- **Rejected**: {evo_summary.get('rejected', 0)}
- **Success Rate**: {evo_summary.get('success_rate', 0):.2%}
- **Rejection Rate**: {evo_summary.get('rejection_rate', 0):.2%}

"""
        
        # Add stability summary
        if "stability_results" in results and "summary" in results["stability_results"]:
            stab_summary = results["stability_results"]["summary"]
            report += f"""### Stability Metrics
- **Mean Perplexity**: {stab_summary.get('mean_perplexity', 0):.2f}
- **Perplexity Std**: {stab_summary.get('std_perplexity', 0):.2f}
- **Mean Consistency**: {stab_summary.get('mean_consistency', 0):.3f}
- **Consistency Std**: {stab_summary.get('std_consistency', 0):.3f}

"""
        
        # Add configuration details
        report += """## Configuration Details

```yaml
"""
        report += yaml.dump(self.base_config, default_flow_style=False)
        report += "```\n"
        
        # Save report to file
        report_path = self.experiment_dir / "experiment_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.success(f"Report generated: {report_path}")
        
        return report


# Example usage
def example_experiment():
    """Example research experiment using the template."""
    
    # Initialize experiment
    experiment = ResearchExperiment(
        experiment_name="basic_evolution_stability",
        config_path="configs/default.yaml"
    )
    
    # Set up engine with custom configuration
    custom_config = {
        "experiment": {
            "device": "cpu",  # Use CPU for reproducibility
        },
        "evolution": {
            "learning_rate": 0.005,  # Slower learning for stability
        }
    }
    
    engine = experiment.setup_engine(custom_config)
    
    # Define test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning involves",
        "Python is a programming language for",
        "Artificial intelligence aims to",
        "Deep learning models can",
    ]
    
    # Run evolution sequence
    evolution_results = experiment.run_evolution_sequence(engine, test_prompts)
    experiment.results["evolution_results"] = evolution_results
    
    # Measure stability
    probe_sentences = [
        "The sun rises in the east.",
        "Water boils at 100 degrees Celsius.",
        "Python supports object-oriented programming.",
    ]
    
    stability_results = experiment.measure_stability(engine, probe_sentences)
    experiment.results["stability_results"] = stability_results
    
    # Save results
    experiment.save_results(experiment.results)
    
    # Generate report
    report = experiment.generate_report(experiment.results)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print(f"Results saved in: {experiment.experiment_dir}")
    print("="*60)
    
    return experiment.results


if __name__ == "__main__":
    # Run example experiment
    results = example_experiment()
    
    # Print summary
    print("\nExperiment Summary:")
    print(json.dumps(results.get("evolution_results", {}).get("summary", {}), indent=2))