"""
Reward computation utilities.
"""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def compute_rewards_batched(
    responses: List[str],
    batch: List[Dict],
    reward_model,
    max_length: int,
    judge_reward_weight: float = 1.0,
    length_reward_weight: float = 0.2
) -> Tuple[List[float], List[float], List[float], List[str], List[int], List[float]]:
    """OPTIMIZATION: Batch compute rewards for all responses at once."""
    rewards = []
    judge_scores = []
    judge_rewards = []
    judge_explanations = []
    tree_walk_lengths = []
    tree_walk_length_rewards = []
    
    # Build all evaluation data - each response gets its OWN prompt/principle
    evaluations = []
    
    for i, (response, ex) in enumerate(zip(responses, batch)):
        calculator = ex["calculator"]
        target = calculator.tree_structure["target"]
        
        # Format given values
        leaves = [
            leaf for leaf in sorted(calculator.tree_structure.get("leaf_nodes", set()))
            if leaf in calculator.values
        ]
        value_lines = []
        for leaf in leaves:
            value = calculator.values[leaf]
            unit = calculator._get_si_unit(leaf)
            value_lines.append(f"{leaf} = {value:.4f}{f' {unit}' if unit else ''}")
        
        values_text = "\n".join(f"  {line}" for line in value_lines) if value_lines else "  (no values found)"
        allowed_vars = ", ".join(leaves) if leaves else "None"
        
        prompt = (
            f"You are scoring candidate word problems that must ask for the value of {target}.\n"
            "The question should be grammatically correct and should be a valid question."
            "A good question must:\n"
            "• Include every given value as provided with its SI unit.\n"
            "• Use only the allowed variables (no intermediate or invented variables).\n"
            f"• End with asking for target variable\n\n"
            f"Allowed variables: {allowed_vars}\n"
            f"Given values:\n{values_text}\n\n"
            f"If the values are not exact then also consider it good\n"
            "Score the question on a scale from 0 to 10, where:\n"
            "- 10 = Perfect: All requirements met perfectly\n"
            "- 7-9 = Good: Minor issues\n"
            "- 4-6 = Acceptable: Some requirements missing\n"
            "- 0-3 = Poor: Major requirements missing\n"
        )
        
        principle = (
            "If the physical scenario in question is not realistic given the variables give it 0 reward. Give Non zero reward only when it's realistic"
            "Score questions on a 0-10 scale. Rank higher (8-10) any question that restates every provided value with its unit, "
            f"sticks to the allowed variable names, stays concise, and ends with asking for value of {target}. "
            "Penalize (lower scores 0-7) missing values, invented variables, and vague language. "
            "Return scores as numeric values between 0 and 10."
        )
        
        evaluations.append({
            'response': response,
            'prompt': prompt,
            'principle': principle,
            'index': i
        })
    
    # CORRECTED: Evaluate each response with its corresponding prompt/principle
    # IMPORTANT: Process evaluations individually to ensure correct matching between
    # each response and its corresponding prompt/principle
    # Batch API may return results out of order, so we use individual evaluation for correctness
    try:
        # Check if reward_model is a Client (vLLM) or local model
        is_client = hasattr(reward_model, 'judge_batch')
        
        # Use individual evaluation to ensure correct matching (batch may have ordering issues)
        # This ensures each response is evaluated with its own prompt/principle
        is_client = False
        
        if is_client:
            # Use batch processing for vLLM client (more efficient)
            logger.debug("Using batch processing for reward evaluation")
            batch_requests = [
                {
                    "principle": eval_data['principle'],
                    "prompt": eval_data['prompt'],
                    "responses": {"response": eval_data['response']}
                }
                for eval_data in evaluations
            ]
            
            try:
                batch_results = reward_model.judge_batch(batch_requests)
                
                # Ensure results match evaluations by index - process in order
                if len(batch_results) != len(evaluations):
                    logger.warning(f"Batch results count ({len(batch_results)}) doesn't match evaluations count ({len(evaluations)}). Falling back to individual evaluation.")
                    is_client = False
                else:
                    # Process results in the same order as evaluations to ensure correct matching
                    for i in range(len(evaluations)):
                        eval_data = evaluations[i]
                        result = batch_results[i]
                        try:
                            if result.scores and "response" in result.scores:
                                raw_score = result.scores["response"]
                                score = max(0.0, min(10.0, raw_score))
                                explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                            else:
                                score = 5.0
                                explanation = f"Score extraction failed. Raw scores: {result.scores}"
                            
                            judge_scores.append(score)
                            judge_explanations.append(explanation)
                            judge_reward = (score / 10.0) * 2.0 - 1.0
                            judge_rewards.append(judge_reward)
                        except Exception as e:
                            logger.error(f"Failed to process batch result {i} (eval index {eval_data['index']}): {e}")
                            judge_scores.append(0.0)
                            judge_explanations.append(f"Batch processing failed: {str(e)}")
                            judge_rewards.append(-1.0)
            except Exception as e:
                logger.error(f"Batch evaluation failed: {e}. Falling back to individual evaluation.")
                is_client = False  # Fall through to individual evaluation
        
        if not is_client:
            # Evaluate each response individually (for local model or fallback)
            for eval_data in evaluations:
                try:
                    result = reward_model.judge(
                        principle=eval_data['principle'],
                        prompt=eval_data['prompt'],
                        responses={"response": eval_data['response']}
                    )
                    
                    if result.scores and "response" in result.scores:
                        raw_score = result.scores["response"]
                        score = max(0.0, min(10.0, raw_score))
                        explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                    else:
                        score = 5.0
                        explanation = f"Score extraction failed. Raw scores: {result.scores}"
                    
                    judge_scores.append(score)
                    judge_explanations.append(explanation)
                    judge_reward = (score / 10.0) * 2.0 - 1.0
                    judge_rewards.append(judge_reward)
                    
                except Exception as e:
                    logger.error(f"Individual evaluation failed for response {eval_data['index']}: {e}")
                    judge_scores.append(0.0)
                    judge_explanations.append(f"Evaluation failed: {str(e)}")
                    judge_rewards.append(-1.0)
    except Exception as e:
        logger.error(f"Batched evaluation failed: {e}. Falling back to individual evaluation.")
        # Fallback: evaluate individually
        for i, (response, ex) in enumerate(zip(responses, batch)):
            try:
                calculator = ex["calculator"]
                target = calculator.tree_structure["target"]
                
                leaves = [
                    leaf for leaf in sorted(calculator.tree_structure.get("leaf_nodes", set()))
                    if leaf in calculator.values
                ]
                value_lines = []
                for leaf in leaves:
                    value = calculator.values[leaf]
                    unit = calculator._get_si_unit(leaf)
                    value_lines.append(f"{leaf} = {value:.4f}{f' {unit}' if unit else ''}")
                
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
                    "Score the question on a scale from 0 to 10."
                )
                
                principle = (
                    "Score questions on a 0-10 scale. Return scores as numeric values between 0 and 10."
                )
                
                result = reward_model.judge(
                    principle=principle,
                    prompt=prompt,
                    responses={"response": response}
                )
                
                if result.scores and "response" in result.scores:
                    raw_score = result.scores["response"]
                    score = max(0.0, min(10.0, raw_score))
                    explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                else:
                    score = 5.0
                    explanation = "Score extraction failed"
                
                judge_scores.append(score)
                judge_explanations.append(explanation)
                judge_reward = (score / 10.0) * 2.0 - 1.0
                judge_rewards.append(judge_reward)
            except Exception as e2:
                logger.error(f"Individual evaluation failed for response {i}: {e2}")
                judge_scores.append(0.0)
                judge_explanations.append(f"Evaluation failed: {str(e2)}")
                judge_rewards.append(-1.0)
    
    # Calculate tree walk length rewards
    for i, ex in enumerate(batch):
        calculator = ex["calculator"]
        tree_walk_length = 0
        tree_walk_length_reward = 0.0
        
        if hasattr(calculator, 'tree_structure') and calculator.tree_structure:
            levels = calculator.tree_structure.get('levels', {})
            leaf_nodes = calculator.tree_structure.get('leaf_nodes', set())
            if levels:
                non_leaf_levels = []
                for level_num, level_nodes in levels.items():
                    non_leaf_in_level = [n for n in level_nodes if n not in leaf_nodes]
                    if non_leaf_in_level:
                        non_leaf_levels.append(level_num)
                
                if non_leaf_levels:
                    max_level = max(non_leaf_levels)
                    tree_walk_length = max_level + 1
                else:
                    tree_walk_length = 1
                
                normalized_length = tree_walk_length / max_length
                tree_walk_length_reward = normalized_length * 2.0 - 1.0
        
        tree_walk_lengths.append(tree_walk_length)
        tree_walk_length_rewards.append(tree_walk_length_reward)
    
    # Combine rewards with weights
    for i in range(len(responses)):
        combined_reward = (
            judge_reward_weight * judge_rewards[i] + 
            length_reward_weight * tree_walk_length_rewards[i]
        )
        rewards.append(combined_reward)
    
    return rewards, judge_scores, judge_rewards, judge_explanations, tree_walk_lengths, tree_walk_length_rewards

