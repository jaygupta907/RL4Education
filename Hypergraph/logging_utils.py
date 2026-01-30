"""
Logging utilities for episode results with hypergraph traces.
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from hypergraph_traverser import HypergraphTraverser

logger = logging.getLogger(__name__)


def format_solution_trace(traverser: HypergraphTraverser, trace: Dict, target: str) -> Dict:
    """
    Format the solution trace from the hypergraph into a structured dictionary.
    
    Args:
        traverser: HypergraphTraverser instance
        trace: Formatted trace from traverser
        target: Target variable
        
    Returns:
        Dictionary containing the solution trace information
    """
    formulas = trace.get('formulas', [])
    leaf_nodes = trace.get('leaf_nodes', [])
    cycle_nodes = trace.get('cycle_nodes', [])
    
    # Collect given values (leaf nodes)
    given_values = []
    for leaf in sorted(leaf_nodes):
        # Get SI unit from hypergraph if available
        # Leaf nodes are INPUT variables, so we need to check input_si_units
        si_unit = ""
        for hyperedge in traverser.hypergraph['hyperedges']:
            input_si_units = hyperedge.get('input_si_units', {})
            if leaf in input_si_units:
                si_unit = input_si_units[leaf]
                break
        
        given_values.append({
            "variable": leaf,
            "unit": si_unit if si_unit else None
        })
    
    # Collect calculation steps
    calculation_steps = []
    for i, formula_info in enumerate(formulas):
        step_info = {
            "step": i + 1,
            "variable": formula_info['output'],
            "formula": formula_info['formula'],
            "inputs": formula_info['inputs'],
            "output_unit": formula_info.get('output_si_unit', ''),
        }
        calculation_steps.append(step_info)
    
    return {
        "target": target,
        "given_values": given_values,
        "calculation_steps": calculation_steps,
        "cycle_nodes": sorted(cycle_nodes),
        "depth": trace.get('depth', 0),
        "num_formulas": len(formulas)
    }


def log_episode_results_sync(
    episode: int,
    responses: List[str],
    rewards: List[float],
    judge_scores: List[float],
    judge_rewards: List[float],
    judge_explanations: List[str],
    batch: List[Dict],
    logs_dir: str,
    config,
):
    """
    Synchronous version of log episode results (for threading).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"episode_{episode+1:04d}_{timestamp}.json"
    log_path = os.path.join(logs_dir, log_filename)
    
    log_data = {
        "episode": episode + 1,
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(responses),
        "questions": []
    }
    
    for i, (response, reward, score, judge_reward, explanation) in enumerate(
        zip(responses, rewards, judge_scores, judge_rewards, judge_explanations)
    ):
        question_data = {
            "question_index": i + 1,
            "question": response,
            "judge_reward": float(judge_reward),
            "judge_score": float(score),
            "explanation": explanation
        }
        
        # Add user prompt if available
        if i < len(batch) and "user_prompt" in batch[i]:
            question_data["user_prompt"] = batch[i]["user_prompt"]
        elif i < len(batch) and "metadata" in batch[i] and "user_prompt" in batch[i]["metadata"]:
            question_data["user_prompt"] = batch[i]["metadata"]["user_prompt"]
        
        if i < len(batch) and "metadata" in batch[i]:
            question_data["metadata"] = batch[i]["metadata"]
        
        # Add solution trace if available
        if i < len(batch) and "traverser" in batch[i] and "trace" in batch[i]:
            traverser = batch[i]["traverser"]
            trace = batch[i]["trace"]
            target = batch[i].get("target", "")
            try:
                solution_trace = format_solution_trace(traverser, trace, target)
                question_data["solution_trace"] = solution_trace
            except Exception as e:
                logger.warning(f"Failed to format solution trace for question {i+1}: {e}")
                question_data["solution_trace"] = {"error": str(e)}
        
        log_data["questions"].append(question_data)
    
    if rewards:
        summary = {
            "judge": {
                "average_score": float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0,
                "min_score": float(min(judge_scores)) if judge_scores else 0.0,
                "max_score": float(max(judge_scores)) if judge_scores else 0.0,
                "average_reward": float(sum(judge_rewards) / len(judge_rewards)) if judge_rewards else 0.0,
                "min_reward": float(min(judge_rewards)) if judge_rewards else 0.0,
                "max_reward": float(max(judge_rewards)) if judge_rewards else 0.0
            }
        }
        
        log_data["summary"] = summary
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Episode results logged to: {log_path}")
    except Exception as e:
        logger.error(f"Failed to write log file {log_path}: {e}")

