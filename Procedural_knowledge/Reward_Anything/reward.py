"""
Evaluate generated word-problem questions with the RewardAnything LLM.

The script mirrors the question-building logic used in finetune_question_generator.py:
it walks the variable dependency graph, selects a target variable, surfaces the leaf
values (dependent variables), and builds candidate questions that ask for the target.
The reward model scores the candidates against a principled rubric.
"""

import os
import rewardanything
from typing import Dict, List, Tuple
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tree_walk_calculation import TreeWalkCalculator

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Paths and knobs
GRAPH_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "variable_concept_graph.json"))
TARGET_VARIABLE = "magnetic_flux"
MAX_LENGTH = 4


def _format_given_values(calculator: TreeWalkCalculator) -> Tuple[List[str], List[str]]:
    """Return ordered leaf variable names and human-readable value strings."""
    leaves = [
        leaf for leaf in sorted(calculator.tree_structure.get("leaf_nodes", set()))
        if leaf in calculator.values
    ]
    value_lines: List[str] = []
    for leaf in leaves:
        value = calculator.values[leaf]
        unit = calculator._get_si_unit(leaf)
        value_lines.append(f"{leaf} = {value:.4f}{f' {unit}' if unit else ''}")
    return leaves, value_lines


def _build_prompt_and_responses(calculator: TreeWalkCalculator) -> Tuple[str, Dict[str, str]]:
    """Construct the evaluation prompt and candidate questions for the reward model."""
    target = calculator.tree_structure["target"]
    leaves, value_lines = _format_given_values(calculator)

    values_text = "\n".join(f"  {line}" for line in value_lines) if value_lines else "  (no values found)"
    allowed_vars = ", ".join(leaves) if leaves else "None"

    prompt = (
        f"You are scoring candidate word problems that must ask for the value of {target}.\n"
        "A good question must:\n"
        "• Include every given value exactly as provided (no rounding) with its SI unit.\n"
        "• Use only the allowed variables (no intermediate or invented variables).\n"
        f"• End with: \"What is the {target}?\"\n\n"
        f"Allowed variables: {allowed_vars}\n"
        f"Given values:\n{values_text}\n\n"
        "Select the question that best follows these requirements."
    )

    values_inline = "; ".join(value_lines) if value_lines else "the provided values"
    responses = {
        # Fully compliant question that mirrors finetune_question_generator expectations.
        "response_a": (
            f"A setup provides {values_inline}. Using only those quantities and no other variables, "
            f"compute the {target}. What is the {target}?"
        ),
        # Omits some values and is slightly vague.
        "response_b": (
            f"Given {value_lines[0] if value_lines else 'one of the values'}, determine the {target}. "
            f"What is the {target}?"
        ),
        # Too vague and missing required structure.
        "response_c": "Find the required result without worrying about the specific numbers."
    }

    return prompt, responses


def main():
    # Prepare dependency tree and sample values for the chosen target variable.
    calculator = TreeWalkCalculator(GRAPH_FILE, max_length=MAX_LENGTH)
    result = calculator.run(TARGET_VARIABLE, min_val=1.0, max_val=100.0)
    if result is None:
        raise RuntimeError(f"Failed to generate a valid tree walk for {TARGET_VARIABLE}")

    prompt, responses = _build_prompt_and_responses(calculator)
    target = calculator.tree_structure["target"]
    principle = (
        "Rank higher any question that restates every provided value with its unit, "
        f"sticks to the allowed variable names, stays concise, and ends with 'What is the {target}?'. "
        "Penalize missing or rounded values, invented variables, and vague language."
    )

    # Load reward model locally (similar to HuggingFace)
    reward_model = rewardanything.from_pretrained(
        "WisdomShell/RewardAnything-8B-v1",
        device=DEVICE,
        torch_dtype="auto"
    )

    # Get comprehensive evaluation
    result = reward_model.judge(
        principle=principle,
        prompt=prompt,
        responses=responses
    )

    print(f"Scores: {result.scores}")
    print(f"Best to worst: {result.ranking}")
    print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
