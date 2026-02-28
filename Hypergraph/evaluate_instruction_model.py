"""
Evaluation script for instruction fine-tuned model.

This script performs hypergraph traversal of different lengths (equally distributed)
and generates questions using just the instruction-tuned model, logging the results.
It also evaluates faithfulness scores using a reward model served via vLLM.
"""
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from hypergraph_traverser import HypergraphTraverser
from prompt_generator import create_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Plotting will be skipped.")

# Try to import rewardanything
try:
    import rewardanything
    REWARDANYTHING_AVAILABLE = True
except ImportError:
    REWARDANYTHING_AVAILABLE = False
    logger.warning("rewardanything library not available. Install with: pip install rewardanything")


class InstructionModelEvaluator:
    """Evaluator for instruction-tuned model with hypergraph traversal."""
    
    def __init__(
        self,
        model_path: str,
        hypergraph_file: str = "formula_hypergraph.json",
        use_quantization: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_vllm_reward: bool = True,
        reward_server_url: str = "http://localhost:8001",
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to instruction-tuned model
            hypergraph_file: Path to formula_hypergraph.json
            use_quantization: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            use_vllm_reward: Whether to use vLLM reward server
            reward_server_url: URL of the reward server
        """
        self.model_path = model_path
        self.hypergraph_file = hypergraph_file
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_vllm_reward = use_vllm_reward
        self.reward_server_url = reward_server_url
        
        # Initialize traverser
        logger.info(f"Loading hypergraph from {hypergraph_file}")
        self.traverser = HypergraphTraverser(hypergraph_file)
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize reward model
        self._load_reward_model()
        
    def _load_model(self):
        """Load the instruction-tuned model and tokenizer."""
        logger.info(f"Loading instruction-tuned model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        
        if self.use_quantization:
            logger.info("Enabling 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        logger.info("Model loaded successfully")
    
    def _load_reward_model(self):
        """Load the reward model for faithfulness scoring."""
        if not REWARDANYTHING_AVAILABLE:
            logger.warning("RewardAnything not available. Faithfulness scoring will be skipped.")
            self.reward_model = None
            return
        
        logger.info("Initializing RewardAnything reward model...")
        try:
            if self.use_vllm_reward:
                # Use vLLM deployment
                logger.info(f"Connecting to RewardAnything server at {self.reward_server_url}")
                self.reward_model = rewardanything.Client(self.reward_server_url)
                logger.info("RewardAnything client connected successfully.")
            else:
                # Use local inference
                logger.info("Loading RewardAnything model locally...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.reward_model = rewardanything.from_pretrained(
                    "zhuohaoyu/RewardAnything-8B-v1",
                    device=str(device),
                    torch_dtype="auto"
                )
                logger.info("RewardAnything reward model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize reward model: {e}")
            if self.use_vllm_reward:
                logger.error("Make sure the RewardAnything server is running.")
            logger.warning("Continuing without reward model. Faithfulness scores will not be computed.")
            self.reward_model = None
    
    def compute_faithfulness_score(
        self,
        generated_question: str,
        prompt: str,
        target: str,
        trace: Dict,
        generated_values: Dict = None
    ) -> Tuple[float, str]:
        """
        Compute faithfulness score for a generated question.
        Simplified to check only:
        1. Given variables are present
        2. Values match
        3. Question asks for target variable
        
        Args:
            generated_question: The generated question text
            prompt: The original prompt used for generation
            target: Target variable
            trace: The hypergraph trace used
            generated_values: Dictionary of generated values for variables
            
        Returns:
            Tuple of (score, explanation) where score is between 1-10
        """
        if self.reward_model is None:
            return None, "Reward model not available"
        
        # Build faithfulness principle - simplified to check only:
        # 1. Given variables are present
        # 2. Values match
        # 3. Question asks for target variable
        leaf_nodes = trace.get('leaf_nodes', [])
        
        # Build list of given variables with their values
        given_vars_with_values = []
        if generated_values:
            for var in sorted(leaf_nodes):
                if var in generated_values:
                    value = generated_values[var].get('value', 'N/A')
                    unit = generated_values[var].get('unit', '')
                    unit_str = f" {unit}" if unit else ""
                    given_vars_with_values.append(f"{var} = {value}{unit_str}")
                else:
                    given_vars_with_values.append(f"{var} = (value not specified)")
        else:
            given_vars_with_values = [var for var in sorted(leaf_nodes)]
        
        given_vars_text = "\n".join(given_vars_with_values)
        num_variables = len(leaf_nodes)
        
        # Calculate thresholds for partial credit
        most_threshold = max(1, int(num_variables * 0.8))  # 80% of variables
        some_threshold = max(1, int(num_variables * 0.5))  # 50% of variables
        
        principle = f"""Score the faithfulness of the generated physics question on a scale of 1-10.

Check these three things:

1. Given variables are present: The question should mention or use these variables:
{given_vars_text}

2. Values match: The question should use the exact same values. Count how many values are present and match:
{chr(10).join(given_vars_with_values)}
Total variables to check: {num_variables}

3. Target variable: The question should ask for: {target}

Scoring Guidelines (give partial credit - missing one value should NOT cause complete failure):
- Score 9-10: All {num_variables} values present AND match exactly AND asks for correct target
- Score 7-8: Most values present (at least {most_threshold} out of {num_variables}) AND values match AND asks for correct target
- Score 5-6: Some values present (at least {some_threshold} out of {num_variables}) AND asks for correct target
- Score 3-4: Few values present (less than {some_threshold}) OR wrong target variable
- Score 1-2: Missing most values AND wrong target variable

IMPORTANT: Missing one or two values should NOT cause complete failure. Give partial credit based on how many values are present and correct.
A question can still be faithful (score 7-8) even if it doesn't include every single value, as long as it includes most of them ({most_threshold}+) and asks for the correct target.

Return scores as numeric values between 1 and 10.
"""
        
        try:
            result = self.reward_model.judge(
                principle=principle,
                prompt=prompt,
                responses={"response": generated_question}
            )
            
            if result.scores and "response" in result.scores:
                raw_score = result.scores["response"]
                # Clamp score between 1-10
                score = max(1.0, min(10.0, raw_score))
                explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                return score, explanation
            else:
                logger.warning(f"Score extraction failed. Raw scores: {result.scores}")
                return 5.0, "Score extraction failed"
                
        except Exception as e:
            logger.error(f"Failed to compute faithfulness score: {e}")
            return None, f"Evaluation failed: {str(e)}"
    
    def get_all_targets(self) -> List[str]:
        """Get all available target variables from the hypergraph."""
        return sorted(self.traverser.all_nodes)
    
    def find_traces_by_length(
        self,
        target: str,
        max_depth: int = 10,
        max_traces: int = 100
    ) -> Dict[int, List[Dict]]:
        """
        Find all traces for a target and group them by trace length.
        
        Args:
            target: Target variable
            max_depth: Maximum depth for traversal
            max_traces: Maximum number of traces to find
            
        Returns:
            Dictionary mapping trace length (number of formulas) to list of traces
        """
        logger.info(f"Finding traces for target: {target}")
        traces = self.traverser.find_all_traces(target, max_depth, max_traces)
        
        # Format traces
        formatted_traces = [self.traverser.format_trace(trace) for trace in traces]
        
        # Group by trace length (number of formulas)
        traces_by_length = defaultdict(list)
        for trace in formatted_traces:
            num_formulas = trace['num_formulas']
            traces_by_length[num_formulas].append(trace)
        
        logger.info(f"Found {len(formatted_traces)} traces for {target}")
        logger.info(f"Trace length distribution: {dict((k, len(v)) for k, v in sorted(traces_by_length.items()))}")
        
        return dict(traces_by_length)
    
    def sample_traces_by_length_range(
        self,
        traces_by_length: Dict[int, List[Dict]],
        min_length: int = 2,
        max_length: int = 10,
        num_samples_per_length: int = 10,
        max_samples_per_length: int = 20
    ) -> List[Dict]:
        """
        Sample traces for specific length range (2-10 by default).
        
        Args:
            traces_by_length: Dictionary mapping length to traces
            min_length: Minimum trace length to include
            max_length: Maximum trace length to include
            num_samples_per_length: Number of traces to sample per length
            max_samples_per_length: Maximum number of samples per length (caps num_samples_per_length)
            
        Returns:
            List of sampled traces
        """
        sampled_traces = []
        
        # Cap num_samples_per_length at max_samples_per_length
        num_samples_per_length = min(num_samples_per_length, max_samples_per_length)
        
        # Only process lengths in the specified range
        for length in range(min_length, max_length + 1):
            if length not in traces_by_length:
                logger.warning(f"No traces found for length {length}")
                continue
            
            traces = traces_by_length[length]
            # Sample exactly num_samples_per_length traces (or all if fewer available)
            # But never exceed max_samples_per_length
            num_to_sample = min(num_samples_per_length, len(traces), max_samples_per_length)
            if num_to_sample == 0:
                logger.warning(f"No traces available for length {length}")
                continue
            
            sampled = random.sample(traces, num_to_sample)
            sampled_traces.extend(sampled)
            logger.info(f"Sampled {num_to_sample} traces of length {length}")
        
        return sampled_traces
    
    def generate_question(self, prompt_text: str) -> str:
        """
        Generate a question using the instruction-tuned model.
        
        Args:
            prompt_text: Formatted prompt text
            
        Returns:
            Generated question text
        """
        # Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response (only the new tokens)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def evaluate(
        self,
        targets: List[str] = None,
        max_depth: int = 10,
        max_traces: int = 100,
        min_length: int = 2,
        max_length: int = 10,
        num_samples_per_length: int = 10,
        num_targets: int = None,
        output_dir: str = "./checkpoints/logs",
        max_samples_per_length: int = 20
    ) -> Dict:
        """
        Run evaluation on multiple targets with traces of specific lengths.
        
        Args:
            targets: List of target variables to evaluate. If None, uses all targets.
            max_depth: Maximum depth for traversal
            max_traces: Maximum number of traces to find per target
            min_length: Minimum trace length to evaluate (default: 2)
            max_length: Maximum trace length to evaluate (default: 10)
            num_samples_per_length: Number of traces to sample per length (default: 10)
            num_targets: Maximum number of targets to evaluate. If None, evaluates all.
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        if targets is None:
            all_targets = self.get_all_targets()
            if num_targets:
                targets = random.sample(all_targets, min(num_targets, len(all_targets)))
            else:
                targets = all_targets
        
        logger.info(f"Evaluating {len(targets)} targets")
        
        # Create output directory and files
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSONL file for incremental saving (one result per line)
        jsonl_path = os.path.join(output_dir, f"instruction_model_evaluation_{timestamp}.jsonl")
        jsonl_file = open(jsonl_path, 'w')
        
        evaluation_results = {
            "timestamp": timestamp,
            "model_path": self.model_path,
            "hypergraph_file": self.hypergraph_file,
            "evaluation_config": {
                "max_depth": max_depth,
                "max_traces": max_traces,
                "min_length": min_length,
                "max_length": max_length,
                "num_samples_per_length": num_samples_per_length,
                "max_samples_per_length": max_samples_per_length,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "use_vllm_reward": self.use_vllm_reward,
                "reward_server_url": self.reward_server_url if self.use_vllm_reward else None,
            },
            "results": []
        }
        
        total_evaluations = 0
        
        logger.info(f"Incremental results will be saved to: {jsonl_path}")
        
        for target_idx, target in enumerate(targets, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating target {target_idx}/{len(targets)}: {target}")
            logger.info(f"{'='*80}")
            
            try:
                # Find traces grouped by length
                traces_by_length = self.find_traces_by_length(
                    target,
                    max_depth=max_depth,
                    max_traces=max_traces
                )
                
                if not traces_by_length:
                    logger.warning(f"No traces found for target: {target}")
                    continue
                
                # Sample traces for specific length range (2-10)
                # Ensure max 20 examples per length
                sampled_traces = self.sample_traces_by_length_range(
                    traces_by_length,
                    min_length=min_length,
                    max_length=max_length,
                    num_samples_per_length=num_samples_per_length,
                    max_samples_per_length=max_samples_per_length
                )
                
                # Evaluate each sampled trace
                for trace_idx, trace in enumerate(sampled_traces, 1):
                    logger.info(f"\n  Trace {trace_idx}/{len(sampled_traces)}: "
                              f"Length={trace['num_formulas']}, Depth={trace['depth']}")
                    
                    try:
                        # Create prompt
                        prompt_text, metadata = create_prompt(
                            self.traverser,
                            trace,
                            target,
                            self.tokenizer
                        )
                        
                        # Generate question
                        generated_question = self.generate_question(prompt_text)
                        
                        # Compute faithfulness score using reward model
                        faithfulness_score = None
                        faithfulness_explanation = None
                        if self.reward_model is not None:
                            logger.info(f"    Computing faithfulness score...")
                            faithfulness_score, faithfulness_explanation = self.compute_faithfulness_score(
                                generated_question=generated_question,
                                prompt=metadata.get('user_prompt', ''),
                                target=target,
                                trace=trace,
                                generated_values=metadata.get('generated_values', {})
                            )
                            if faithfulness_score is not None:
                                logger.info(f"    Faithfulness score: {faithfulness_score:.2f}/10")
                        
                        # Store detailed result with all information
                        trace_length = trace['num_formulas']
                        leaf_nodes = trace.get('leaf_nodes', [])
                        generated_values = metadata.get('generated_values', {})
                        
                        # Build given variables with values
                        given_variables = {}
                        for var in leaf_nodes:
                            if var in generated_values:
                                given_variables[var] = {
                                    "value": generated_values[var].get("value"),
                                    "unit": generated_values[var].get("unit", "")
                                }
                            else:
                                given_variables[var] = {"value": None, "unit": ""}
                        
                        detailed_result = {
                            "length": trace_length,
                            "score": faithfulness_score,
                            "question": generated_question,
                            "target_variable": target,
                            "given_variables": given_variables,
                            "faithfulness_explanation": faithfulness_explanation,
                            "trace_depth": trace['depth'],
                            "formulas": trace['formulas'],
                            "calculation_steps": trace.get('calculation_steps', []),
                        }
                        
                        # Save detailed result immediately to JSONL file
                        jsonl_file.write(json.dumps(detailed_result) + '\n')
                        jsonl_file.flush()  # Ensure it's written to disk immediately
                        
                        # Also store full result for statistics
                        full_result = {
                            "target": target,
                            "trace_length": trace_length,
                            "trace_depth": trace['depth'],
                            "leaf_nodes": trace['leaf_nodes'],
                            "cycle_nodes": trace['cycle_nodes'],
                            "formulas": trace['formulas'],
                            "prompt": metadata.get('user_prompt', ''),
                            "generated_question": generated_question,
                            "generated_values": metadata.get('generated_values', {}),
                            "faithfulness_score": faithfulness_score,
                            "faithfulness_explanation": faithfulness_explanation,
                        }
                        evaluation_results["results"].append(full_result)
                        total_evaluations += 1
                        
                        logger.info(f"    Length={trace_length}, Score={faithfulness_score:.2f if faithfulness_score else 'N/A'}/10")
                        
                    except Exception as e:
                        logger.error(f"    Error evaluating trace {trace_idx}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error evaluating target {target}: {e}")
                continue
        
        # Close JSONL file
        jsonl_file.close()
        logger.info(f"\nIncremental results saved to: {jsonl_path}")
        
        evaluation_results["total_evaluations"] = total_evaluations
        evaluation_results["jsonl_path"] = jsonl_path
        
        # Compute statistics on faithfulness scores
        faithfulness_scores = [
            r["faithfulness_score"] 
            for r in evaluation_results["results"] 
            if r.get("faithfulness_score") is not None
        ]
        
        if faithfulness_scores:
            evaluation_results["faithfulness_statistics"] = {
                "mean": sum(faithfulness_scores) / len(faithfulness_scores),
                "min": min(faithfulness_scores),
                "max": max(faithfulness_scores),
                "count": len(faithfulness_scores),
            }
            logger.info(f"\nFaithfulness Score Statistics:")
            logger.info(f"  Mean: {evaluation_results['faithfulness_statistics']['mean']:.2f}/10")
            logger.info(f"  Min: {evaluation_results['faithfulness_statistics']['min']:.2f}/10")
            logger.info(f"  Max: {evaluation_results['faithfulness_statistics']['max']:.2f}/10")
            logger.info(f"  Evaluated: {evaluation_results['faithfulness_statistics']['count']}/{total_evaluations}")
            
            # Compute statistics by trace length
            scores_by_length = defaultdict(list)
            for result in evaluation_results["results"]:
                trace_length = result.get("trace_length", 0)
                faithfulness_score = result.get("faithfulness_score")
                if faithfulness_score is not None:
                    scores_by_length[trace_length].append(faithfulness_score)
            
            faithfulness_by_length = {}
            for length in sorted(scores_by_length.keys()):
                scores = scores_by_length[length]
                faithfulness_by_length[length] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }
            
            evaluation_results["faithfulness_by_length"] = faithfulness_by_length
            
            logger.info(f"\nFaithfulness Score by Traversal Length:")
            for length in sorted(faithfulness_by_length.keys()):
                stats = faithfulness_by_length[length]
                logger.info(f"  Length {length}: Mean={stats['mean']:.2f}, "
                          f"Min={stats['min']:.2f}, Max={stats['max']:.2f}, "
                          f"Count={stats['count']}")
        else:
            evaluation_results["faithfulness_statistics"] = None
            evaluation_results["faithfulness_by_length"] = None
            logger.warning("No faithfulness scores computed (reward model not available)")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluation completed. Total evaluations: {total_evaluations}")
        logger.info(f"{'='*80}")
        
        return evaluation_results
    
    def plot_faithfulness_by_length(
        self,
        results: Dict,
        output_dir: str = "./checkpoints/logs"
    ) -> Optional[str]:
        """
        Create a plot of average faithfulness score vs traversal length.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save the plot
            
        Returns:
            Path to saved plot file, or None if plotting failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available. Skipping plot generation.")
            return None
        
        # Group scores by trace length
        scores_by_length = defaultdict(list)
        for result in results.get("results", []):
            trace_length = result.get("trace_length", 0)
            faithfulness_score = result.get("faithfulness_score")
            if faithfulness_score is not None:
                scores_by_length[trace_length].append(faithfulness_score)
        
        if not scores_by_length:
            logger.warning("No faithfulness scores available for plotting.")
            return None
        
        # Calculate average scores for each length
        lengths = sorted(scores_by_length.keys())
        avg_scores = [
            sum(scores_by_length[length]) / len(scores_by_length[length])
            for length in lengths
        ]
        counts = [len(scores_by_length[length]) for length in lengths]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(lengths, avg_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Add count annotations
        for i, (length, avg_score, count) in enumerate(zip(lengths, avg_scores, counts)):
            plt.annotate(
                f'n={count}',
                xy=(length, avg_score),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        plt.xlabel('Traversal Length (Number of Formulas)', fontsize=12, fontweight='bold')
        plt.ylabel('Average Faithfulness Score', fontsize=12, fontweight='bold')
        plt.title('Average Faithfulness Score vs Traversal Length', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 10)
        
        # Add horizontal line at y=5 for reference
        plt.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Midpoint (5.0)')
        plt.legend()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        timestamp = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        plot_path = os.path.join(output_dir, f"faithfulness_by_length_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to: {plot_path}")
        return plot_path
    
    def save_results(self, results: Dict, output_dir: str = "./checkpoints/logs"):
        """
        Save final evaluation results summary to a JSON file and create plots.
        Note: Individual results are already saved incrementally to JSONL file.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = results["timestamp"]
        output_path = os.path.join(output_dir, f"instruction_model_evaluation_{timestamp}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Final summary saved to: {output_path}")
        
        # Create and save plot
        plot_path = self.plot_faithfulness_by_length(results, output_dir)
        if plot_path:
            results["plot_path"] = plot_path
        
        return output_path


def main():
    """Main entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate instruction-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to instruction-tuned model"
    )
    parser.add_argument(
        "--hypergraph_file",
        type=str,
        default="formula_hypergraph.json",
        help="Path to formula_hypergraph.json"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum depth for hypergraph traversal"
    )
    parser.add_argument(
        "--max_traces",
        type=int,
        default=100,
        help="Maximum number of traces to find per target"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=2,
        help="Minimum trace length to evaluate (default: 2)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10,
        help="Maximum trace length to evaluate (default: 10)"
    )
    parser.add_argument(
        "--num_samples_per_length",
        type=int,
        default=10,
        help="Number of traces to sample per length (default: 10, max: 20)"
    )
    parser.add_argument(
        "--max_samples_per_length",
        type=int,
        default=20,
        help="Maximum number of samples per length (default: 20)"
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=None,
        help="Maximum number of targets to evaluate (None = all)"
    )
    parser.add_argument(
        "--use_quantization",
        default=False,
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--use_vllm_reward",
        action="store_true",
        default=True,
        help="Use vLLM reward server (default: True)"
    )
    parser.add_argument(
        "--no_vllm_reward",
        dest="use_vllm_reward",
        action="store_false",
        help="Disable vLLM reward server and use local model"
    )
    parser.add_argument(
        "--reward_server_url",
        type=str,
        default="http://localhost:8001",
        help="URL of the reward server (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/logs",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = InstructionModelEvaluator(
        model_path=args.model_path,
        hypergraph_file=args.hypergraph_file,
        use_quantization=args.use_quantization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm_reward=args.use_vllm_reward,
        reward_server_url=args.reward_server_url,
    )
    
    # Run evaluation (results are saved incrementally during evaluation)
    # Ensure max_samples_per_length is used
    results = evaluator.evaluate(
        targets=None,
        max_depth=args.max_depth,
        max_traces=args.max_traces,
        min_length=args.min_length,
        max_length=args.max_length,
        num_samples_per_length=args.num_samples_per_length,
        num_targets=args.num_targets,
        output_dir=args.output_dir,
        max_samples_per_length=args.max_samples_per_length
    )
    
    # Save final summary with statistics and plots
    evaluator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
