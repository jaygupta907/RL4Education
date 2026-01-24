"""
Logging utilities for episode results.
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from tree_walk_calculation import TreeWalkCalculator

logger = logging.getLogger(__name__)


def format_solution_trace(calculator: TreeWalkCalculator) -> Dict:
    """
    Format the solution trace from the calculator into a structured dictionary.
    
    Args:
        calculator: TreeWalkCalculator instance with completed calculation
        
    Returns:
        Dictionary containing the solution trace information
    """
    target = calculator.tree_structure['target']
    target_value = calculator.values.get(target, None)
    
    if target_value is None:
        return {
            "target": target,
            "final_answer": None,
            "given_values": [],
            "calculation_steps": []
        }
    
    # Collect given values (leaf nodes)
    given_values = []
    for leaf in sorted(calculator.tree_structure['leaf_nodes']):
        if leaf in calculator.values:
            si_unit = calculator._get_si_unit(leaf)
            given_values.append({
                "variable": leaf,
                "value": float(calculator.values[leaf]),
                "unit": si_unit if si_unit else None
            })
    
    # Collect calculation steps by level
    calculation_steps = []
    all_levels = sorted(calculator.tree_structure['levels'].keys())
    for level in all_levels:
        if level == 0:  # Skip target level (will be shown separately)
            continue
        level_nodes = calculator.tree_structure['levels'].get(level, [])
        for node in sorted(level_nodes):
            if node not in calculator.tree_structure['leaf_nodes'] and node in calculator.values:
                step_info = {
                    "level": level,
                    "variable": node,
                    "value": float(calculator.values[node])
                }
                
                # Add formula if available
                if 'node_formulas' in calculator.tree_structure:
                    if node in calculator.tree_structure['node_formulas']:
                        formula, deps = calculator.tree_structure['node_formulas'][node]
                        if formula:
                            step_info["formula"] = formula
                            step_info["dependencies"] = sorted(list(deps))
                
                calculation_steps.append(step_info)
    
    return {
        "target": target,
        "final_answer": float(target_value),
        "given_values": given_values,
        "calculation_steps": calculation_steps
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
    tree_walk_lengths: Optional[List[int]] = None,
    tree_walk_length_rewards: Optional[List[float]] = None
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
        
        if tree_walk_lengths and i < len(tree_walk_lengths):
            tree_walk_length = tree_walk_lengths[i]
            question_data["tree_walk_length"] = tree_walk_length
            question_data["max_tree_walk_length"] = config.max_length
            
            if tree_walk_length_rewards and i < len(tree_walk_length_rewards):
                question_data["tree_walk_length_reward"] = float(tree_walk_length_rewards[i])
            else:
                normalized_length = tree_walk_length / config.max_length if config.max_length > 0 else 0.0
                question_data["tree_walk_length_reward"] = float(normalized_length * 2.0 - 1.0)
        
        if i < len(batch) and "metadata" in batch[i]:
            question_data["metadata"] = batch[i]["metadata"]
        
        # Add solution trace if calculator is available
        if i < len(batch) and "calculator" in batch[i]:
            calculator = batch[i]["calculator"]
            try:
                solution_trace = format_solution_trace(calculator)
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
        
        if tree_walk_lengths:
            summary["tree_walk_length"] = {
                "average_length": float(sum(tree_walk_lengths) / len(tree_walk_lengths)),
                "min_length": int(min(tree_walk_lengths)),
                "max_length": int(max(tree_walk_lengths)),
                "max_possible_length": config.max_length
            }
            
            if tree_walk_length_rewards:
                summary["tree_walk_length"]["average_reward"] = float(sum(tree_walk_length_rewards) / len(tree_walk_length_rewards))
                summary["tree_walk_length"]["min_reward"] = float(min(tree_walk_length_rewards))
                summary["tree_walk_length"]["max_reward"] = float(max(tree_walk_length_rewards))
        
        log_data["summary"] = summary
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Episode results logged to: {log_path}")
    except Exception as e:
        logger.error(f"Failed to write log file {log_path}: {e}")

