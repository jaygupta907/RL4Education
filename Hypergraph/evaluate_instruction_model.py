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
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import get_default_config
from hypergraph_traverser import HypergraphTraverser
from prompt_generator import create_prompt
from reward_computer import compute_faithfulness_scores
from utils import clean_decoded_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Plotting will be skipped.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. WandB plot logging will use the static image fallback.")

try:
    import rewardanything
    REWARDANYTHING_AVAILABLE = True
except ImportError:
    REWARDANYTHING_AVAILABLE = False
    logger.warning("rewardanything library not available. Install with: pip install rewardanything")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


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
        batch_size: int = 4,
        use_vllm_reward: bool = True,
        reward_server_url: str = "http://localhost:8001",
    ):
        self.model_path = model_path
        self.hypergraph_file = hypergraph_file
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.use_vllm_reward = use_vllm_reward
        self.reward_server_url = reward_server_url

        logger.info(f"Loading hypergraph from {hypergraph_file}")
        self.traverser = HypergraphTraverser(hypergraph_file)
        self._load_model()
        self._load_reward_model()

    def _load_model(self):
        """Load the instruction-tuned model and tokenizer."""
        logger.info(f"Loading instruction-tuned model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

        if self.use_quantization:
            logger.info("Enabling 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            )
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
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
                logger.info(f"Connecting to RewardAnything server at {self.reward_server_url}")
                self.reward_model = rewardanything.Client(self.reward_server_url)
                logger.info("RewardAnything client connected successfully.")
            else:
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
    ) -> Tuple[Optional[float], str]:
        """Compute the reward-model faithfulness score for a generated question."""
        if self.reward_model is None:
            return None, "Reward model not available"

        leaf_nodes = trace.get('leaf_nodes', [])

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
        most_threshold = max(1, int(num_variables * 0.8))
        some_threshold = max(1, int(num_variables * 0.5))

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
                score = max(1.0, min(10.0, raw_score))
                explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                return score, explanation

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
        """Find all traces for a target and group them by trace length."""
        logger.info(f"Finding traces for target: {target}")
        traces = self.traverser.find_all_traces(target, max_depth, max_traces)
        formatted_traces = [self.traverser.format_trace(trace) for trace in traces]

        traces_by_length = defaultdict(list)
        for trace in formatted_traces:
            traces_by_length[trace['num_formulas']].append(trace)

        logger.info(f"Found {len(formatted_traces)} traces for {target}")
        logger.info(
            f"Trace length distribution: {dict((k, len(v)) for k, v in sorted(traces_by_length.items()))}"
        )
        return dict(traces_by_length)

    def sample_traces_by_length_range(
        self,
        traces_by_length: Dict[int, List[Dict]],
        min_length: int = 2,
        max_length: int = 8,
        num_samples_per_length: int = 20,
        max_samples_per_length: int = 20,
        completed_samples_by_length: Optional[Dict[int, int]] = None,
    ) -> List[Dict]:
        """Sample traces for a specific length range."""
        sampled_traces = []
        num_samples_per_length = min(num_samples_per_length, max_samples_per_length)
        completed_samples_by_length = completed_samples_by_length or {}

        for length in range(min_length, max_length + 1):
            if length not in traces_by_length:
                logger.warning(f"No traces found for length {length}")
                continue

            completed_count = completed_samples_by_length.get(length, 0)
            remaining_slots = max(0, max_samples_per_length - completed_count)
            if remaining_slots == 0:
                logger.info(f"Skipping length {length}: already reached max_samples_per_length={max_samples_per_length}")
                continue

            traces = traces_by_length[length]
            num_to_sample = min(num_samples_per_length, len(traces), remaining_slots)
            if num_to_sample == 0:
                logger.warning(f"No traces available for length {length}")
                continue

            sampled_traces.extend(random.sample(traces, num_to_sample))
            logger.info(
                f"Sampled {num_to_sample} traces of length {length} "
                f"(completed={completed_count}, remaining_after_sample={remaining_slots - num_to_sample})"
            )

        return sampled_traces

    def generate_questions_batched(self, prompt_texts: List[str]) -> List[str]:
        """Generate questions for a batch of prompts using the same pattern as RL training."""
        if not prompt_texts:
            return []

        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

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

        generated_questions = []
        for output_ids, prompt_length in zip(outputs, prompt_lengths):
            generated_ids = output_ids[int(prompt_length):]
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_questions.append(clean_decoded_text(decoded_text))

        return generated_questions

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
        """Run evaluation on multiple targets with traces of specific lengths."""
        if targets is None:
            all_targets = self.get_all_targets()
            if num_targets:
                targets = random.sample(all_targets, min(num_targets, len(all_targets)))
            else:
                targets = all_targets

        logger.info(f"Evaluating {len(targets)} targets")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(output_dir, f"instruction_model_evaluation_{timestamp}.jsonl")

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
                "batch_size": self.batch_size,
                "use_quantization": self.use_quantization,
                "use_vllm_reward": self.use_vllm_reward,
                "reward_server_url": self.reward_server_url if self.use_vllm_reward else None,
            },
            "results": []
        }

        total_evaluations = 0
        completed_samples_by_length = {length: 0 for length in range(min_length, max_length + 1)}

        def reached_max_samples_for_all_lengths() -> bool:
            return all(
                completed_samples_by_length[length] >= max_samples_per_length
                for length in range(min_length, max_length + 1)
            )

        logger.info(f"Incremental results will be saved to: {jsonl_path}")

        with open(jsonl_path, 'w') as jsonl_file:
            for target_idx, target in enumerate(targets, 1):
                if reached_max_samples_for_all_lengths():
                    logger.info(
                        "Stopping evaluation early because all trace lengths reached "
                        f"max_samples_per_length={max_samples_per_length}"
                    )
                    break
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Evaluating target {target_idx}/{len(targets)}: {target}")
                logger.info(f"{'=' * 80}")

                try:
                    traces_by_length = self.find_traces_by_length(
                        target,
                        max_depth=max_depth,
                        max_traces=max_traces
                    )
                    if not traces_by_length:
                        logger.warning(f"No traces found for target: {target}")
                        continue

                    sampled_traces = self.sample_traces_by_length_range(
                        traces_by_length,
                        min_length=min_length,
                        max_length=max_length,
                        num_samples_per_length=num_samples_per_length,
                        max_samples_per_length=max_samples_per_length,
                        completed_samples_by_length=completed_samples_by_length,
                    )

                    batched_examples = []
                    prompt_texts = []
                    for trace_idx, trace in enumerate(sampled_traces, 1):
                        trace_length = trace['num_formulas']
                        if completed_samples_by_length.get(trace_length, 0) >= max_samples_per_length:
                            logger.info(
                                f"Skipping trace of length {trace_length}: "
                                f"already reached max_samples_per_length={max_samples_per_length}"
                            )
                            continue

                        try:
                            prompt_text, metadata = create_prompt(
                                self.traverser,
                                trace,
                                target,
                                self.tokenizer
                            )
                        except Exception as e:
                            logger.error(f"    Error preparing trace {trace_idx}: {e}")
                            continue

                        batched_examples.append({
                            "trace_idx": trace_idx,
                            "trace": trace,
                            "trace_length": trace_length,
                            "metadata": metadata,
                            "query": metadata.get('user_prompt', ''),
                            "target": target,
                        })
                        prompt_texts.append(prompt_text)

                    if not batched_examples:
                        continue

                    for batch_start in range(0, len(batched_examples), self.batch_size):
                        batch_examples = batched_examples[batch_start:batch_start + self.batch_size]
                        batch_prompt_texts = prompt_texts[batch_start:batch_start + self.batch_size]

                        logger.info(
                            f"Generating batch {batch_start // self.batch_size + 1} "
                            f"with {len(batch_examples)} questions (batch_size={self.batch_size})..."
                        )
                        generated_questions = self.generate_questions_batched(batch_prompt_texts)
                        if len(generated_questions) != len(batch_examples):
                            logger.error(
                                f"Batch generation size mismatch: prompts={len(batch_examples)}, "
                                f"responses={len(generated_questions)}"
                            )
                            continue

                        reward_batch = [
                            {
                                "query": example["query"],
                                "trace": example["trace"],
                                "target": example["target"],
                                "metadata": example["metadata"],
                            }
                            for example in batch_examples
                        ]

                        faithfulness_results = [(None, "Reward model not available")] * len(batch_examples)
                        if self.reward_model is not None:
                            logger.info(
                                f"Computing faithfulness for {len(batch_examples)} questions "
                                f"(batch_size={self.batch_size})..."
                            )
                            faithfulness_results = compute_faithfulness_scores(
                                generated_questions,
                                reward_batch,
                                self.reward_model,
                            )

                        for example, generated_question, faithfulness_result in zip(
                            batch_examples,
                            generated_questions,
                            faithfulness_results,
                        ):
                            trace = example["trace"]
                            trace_length = example["trace_length"]
                            metadata = example["metadata"]
                            faithfulness_score, faithfulness_explanation = faithfulness_result

                            if faithfulness_score is not None:
                                logger.info(
                                    f"    Trace {example['trace_idx']}: Length={trace_length}, "
                                    f"Faithfulness score: {faithfulness_score:.2f}/10"
                                )
                            else:
                                logger.info(
                                    f"    Trace {example['trace_idx']}: Length={trace_length}, Faithfulness score: N/A"
                                )

                            leaf_nodes = trace.get('leaf_nodes', [])
                            generated_values = metadata.get('generated_values', {})

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
                            jsonl_file.write(json.dumps(detailed_result) + '\n')
                            jsonl_file.flush()

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
                            completed_samples_by_length[trace_length] = completed_samples_by_length.get(trace_length, 0) + 1

                            score_text = f"{faithfulness_score:.2f}/10" if faithfulness_score is not None else "N/A"
                            logger.info(
                                f"    Length={trace_length}, Score={score_text}, "
                                f"completed_for_length={completed_samples_by_length[trace_length]}/{max_samples_per_length}"
                            )

                            if reached_max_samples_for_all_lengths():
                                logger.info(
                                    "All trace lengths reached the requested max_samples_per_length. "
                                    "Ending evaluation."
                                )
                                break

                        if reached_max_samples_for_all_lengths():
                            break

                    if reached_max_samples_for_all_lengths():
                        break
                except Exception as e:
                    logger.error(f"Error evaluating target {target}: {e}")
                    continue

        logger.info(f"\nIncremental results saved to: {jsonl_path}")
        evaluation_results["total_evaluations"] = total_evaluations
        evaluation_results["jsonl_path"] = jsonl_path
        evaluation_results["completed_samples_by_length"] = completed_samples_by_length

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
            logger.info("\nFaithfulness Score Statistics:")
            logger.info(f"  Mean: {evaluation_results['faithfulness_statistics']['mean']:.2f}/10")
            logger.info(f"  Min: {evaluation_results['faithfulness_statistics']['min']:.2f}/10")
            logger.info(f"  Max: {evaluation_results['faithfulness_statistics']['max']:.2f}/10")
            logger.info(
                f"  Evaluated: {evaluation_results['faithfulness_statistics']['count']}/{total_evaluations}"
            )

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
            logger.info("\nFaithfulness Score by Traversal Length:")
            for length in sorted(faithfulness_by_length.keys()):
                stats = faithfulness_by_length[length]
                logger.info(
                    f"  Length {length}: Mean={stats['mean']:.2f}, "
                    f"Min={stats['min']:.2f}, Max={stats['max']:.2f}, Count={stats['count']}"
                )
        else:
            evaluation_results["faithfulness_statistics"] = None
            evaluation_results["faithfulness_by_length"] = None
            logger.warning("No faithfulness scores computed (reward model not available)")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluation completed. Total evaluations: {total_evaluations}")
        logger.info(f"{'=' * 80}")
        return evaluation_results

    def plot_faithfulness_by_length(
        self,
        results: Dict,
        output_dir: str = "./checkpoints/logs"
    ) -> Optional[str]:
        """Create a plot of average faithfulness score vs traversal length."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available. Skipping plot generation.")
            return None

        scores_by_length = defaultdict(list)
        for result in results.get("results", []):
            trace_length = result.get("trace_length", 0)
            faithfulness_score = result.get("faithfulness_score")
            if faithfulness_score is not None:
                scores_by_length[trace_length].append(faithfulness_score)

        if not scores_by_length:
            logger.warning("No faithfulness scores available for plotting.")
            return None

        lengths = sorted(scores_by_length.keys())
        avg_scores = [sum(scores_by_length[length]) / len(scores_by_length[length]) for length in lengths]
        counts = [len(scores_by_length[length]) for length in lengths]

        plt.figure(figsize=(10, 6))
        plt.plot(lengths, avg_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
        for length, avg_score, count in zip(lengths, avg_scores, counts):
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
        plt.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Midpoint (5.0)')
        plt.legend()

        os.makedirs(output_dir, exist_ok=True)
        timestamp = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        plot_path = os.path.join(output_dir, f"faithfulness_by_length_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Plot saved to: {plot_path}")
        return plot_path

    def build_plotly_faithfulness_by_length_plot(self, results: Dict):
        """Build an interactive Plotly plot for WandB logging."""
        if not PLOTLY_AVAILABLE:
            return None

        scores_by_length = defaultdict(list)
        for result in results.get("results", []):
            trace_length = result.get("trace_length", 0)
            faithfulness_score = result.get("faithfulness_score")
            if faithfulness_score is not None:
                scores_by_length[trace_length].append(faithfulness_score)

        if not scores_by_length:
            logger.warning("No faithfulness scores available for Plotly logging.")
            return None

        lengths = sorted(scores_by_length.keys())
        avg_scores = [sum(scores_by_length[length]) / len(scores_by_length[length]) for length in lengths]
        counts = [len(scores_by_length[length]) for length in lengths]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=lengths,
                y=avg_scores,
                mode="lines+markers+text",
                text=[f"n={count}" for count in counts],
                textposition="top center",
                name="Average faithfulness",
                hovertemplate="Length=%{x}<br>Avg score=%{y:.2f}<br>%{text}<extra></extra>",
            )
        )
        fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Midpoint (5.0)")
        fig.update_layout(
            title="Average Faithfulness Score vs Traversal Length",
            xaxis_title="Traversal Length (Number of Formulas)",
            yaxis_title="Average Faithfulness Score",
            yaxis=dict(range=[0, 10]),
            template="plotly_white",
        )
        return fig

    def save_results(self, results: Dict, output_dir: str = "./checkpoints/logs") -> str:
        """Save final evaluation results summary to a JSON file and create plots."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = results["timestamp"]
        plot_path = self.plot_faithfulness_by_length(results, output_dir)
        if plot_path:
            results["plot_path"] = plot_path

        output_path = os.path.join(output_dir, f"instruction_model_evaluation_{timestamp}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Final summary saved to: {output_path}")
        return output_path


def log_results_to_wandb(run, results: Dict, summary_path: str):
    """Log only the final Plotly plot to Weights & Biases."""
    if PLOTLY_AVAILABLE:
        plotly_fig = InstructionModelEvaluator.build_plotly_faithfulness_by_length_plot(None, results)
        if plotly_fig is not None:
            run.log({"evaluation/faithfulness_by_length_plot": plotly_fig})


def main():
    """Main entry point for evaluation."""
    import argparse

    rl_config = get_default_config()
    default_output_dir = os.path.join(rl_config.logs_dir, "instruction_eval")

    parser = argparse.ArgumentParser(description="Evaluate instruction-tuned model")
    parser.add_argument("--model_path", type=str,default=rl_config.instruction_tuned_model_path,help="Path to istruction-tuned model")
    parser.add_argument(
        "--hypergraph_file",
        type=str,
        default=rl_config.hypergraph_file,
        help="Path to formula_hypergraph.json"
    )
    parser.add_argument("--max_depth", type=int, default=rl_config.max_depth, help="Maximum depth for hypergraph traversal")
    parser.add_argument("--max_traces", type=int, default=rl_config.max_traces, help="Maximum number of traces to find per target")
    parser.add_argument(
        "--min_length",
        type=int,
        default=rl_config.min_trace_length,
        help="Minimum trace length to evaluate"
    )
    parser.add_argument("--max_length", type=int, default=10, help="Maximum trace length to evaluate")
    parser.add_argument(
        "--num_samples_per_length",
        type=int,
        default=10,
        help="Number of traces to sample per length (max 20)"
    )
    parser.add_argument(
        "--max_samples_per_length",
        type=int,
        default=20,
        help="Maximum number of samples per length"
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=None,
        help="Maximum number of targets to evaluate (None = all)"
    )
    parser.add_argument(
        "--use_quantization",
        dest="use_quantization",
        action="store_true",
        default=rl_config.use_quantization,
        help="Use the same quantization setting as RL training"
    )
    parser.add_argument(
        "--no_quantization",
        dest="use_quantization",
        action="store_false",
        help="Disable quantization for evaluation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=rl_config.max_new_tokens,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=rl_config.temperature,
        help="Generation temperature"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=rl_config.batch_size,
        help="Fixed evaluation batch size"
    )
    parser.add_argument("--top_p", type=float, default=rl_config.top_p, help="Top-p sampling parameter")
    parser.add_argument(
        "--use_vllm_reward",
        dest="use_vllm_reward",
        action="store_true",
        default=rl_config.use_vllm_reward,
        help="Use the RL reward-server setting"
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
        default=rl_config.reward_server_url,
        help="URL of the reward server"
    )
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save results")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=rl_config.wandb_project,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=f"{rl_config.experiment_name}-instruction-eval",
        help="Weights & Biases run/group name"
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    args = parser.parse_args()

    wandb_run = None
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            group=rl_config.experiment_name,
            config={
                "evaluation_type": "instruction_tuned_model",
                "model_path": args.model_path,
                "hypergraph_file": args.hypergraph_file,
                "max_depth": args.max_depth,
                "max_traces": args.max_traces,
                "min_length": args.min_length,
                "max_length": args.max_length,
                "num_samples_per_length": args.num_samples_per_length,
                "max_samples_per_length": args.max_samples_per_length,
                "num_targets": args.num_targets,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "batch_size": args.batch_size,
                "use_quantization": args.use_quantization,
                "use_vllm_reward": args.use_vllm_reward,
                "reward_server_url": args.reward_server_url,
                "output_dir": args.output_dir,
                "rl_training_defaults": {
                    "max_depth": rl_config.max_depth,
                    "max_traces": rl_config.max_traces,
                    "min_trace_length": rl_config.min_trace_length,
                    "max_new_tokens": rl_config.max_new_tokens,
                    "temperature": rl_config.temperature,
                    "top_p": rl_config.top_p,
                    "use_quantization": rl_config.use_quantization,
                    "use_vllm_reward": rl_config.use_vllm_reward,
                    "reward_server_url": rl_config.reward_server_url,
                },
            }
        )
        logger.info("Wandb initialized for evaluation logging.")

    evaluator = InstructionModelEvaluator(
        model_path=args.model_path,
        hypergraph_file=args.hypergraph_file,
        use_quantization=args.use_quantization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        use_vllm_reward=args.use_vllm_reward,
        reward_server_url=args.reward_server_url,
    )

    try:
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
        summary_path = evaluator.save_results(results, args.output_dir)

        if wandb_run is not None:
            log_results_to_wandb(wandb_run, results, summary_path)
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
