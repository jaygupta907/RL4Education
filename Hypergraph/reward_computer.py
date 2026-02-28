"""
Reward computation utilities.
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _build_faithfulness_principle(trace: Dict, target: str, generated_values: Optional[Dict] = None) -> str:
    """Build the principle text for faithfulness scoring (same logic as evaluate_instruction_model)."""
    leaf_nodes = trace.get("leaf_nodes", [])
    given_vars_with_values = []
    if generated_values:
        for var in sorted(leaf_nodes):
            if var in generated_values:
                val = generated_values[var]
                value = val.get("value", "N/A") if isinstance(val, dict) else "N/A"
                unit = val.get("unit", "") if isinstance(val, dict) else ""
                unit_str = f" {unit}" if unit else ""
                given_vars_with_values.append(f"{var} = {value}{unit_str}")
            else:
                given_vars_with_values.append(f"{var} = (value not specified)")
    else:
        given_vars_with_values = [f"{v}=?" for v in sorted(leaf_nodes)]
    given_vars_text = "\n".join(given_vars_with_values)
    num_variables = len(leaf_nodes)
    most_threshold = max(1, int(num_variables * 0.8))
    some_threshold = max(1, int(num_variables * 0.5))
    return f"""Score the faithfulness of the generated physics question on a scale of 1-10.

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
Return scores as numeric values between 1 and 10.
"""


def compute_faithfulness_scores(
    responses: List[str],
    batch: List[Dict],
    reward_model,
) -> List[Tuple[Optional[float], Optional[str]]]:
    """
    Compute faithfulness score (1-10) for each generated question.
    Returns list of (score, explanation) per response; (None, None) on failure.
    """
    results: List[Tuple[Optional[float], Optional[str]]] = []
    for i, (response, ex) in enumerate(zip(responses, batch)):
        prompt = ex.get("query", "")
        trace = ex.get("trace", {})
        target = ex.get("target", "")
        metadata = ex.get("metadata") or {}
        generated_values = metadata.get("generated_values")
        if not prompt or not trace:
            logger.warning(f"Faithfulness: missing prompt or trace for item {i}, skipping")
            results.append((None, None))
            continue
        principle = _build_faithfulness_principle(trace, target, generated_values)
        try:
            result = reward_model.judge(
                principle=principle,
                prompt=prompt,
                responses={"response": response},
            )
            if result.scores and "response" in result.scores:
                raw = result.scores["response"]
                score = max(1.0, min(10.0, float(raw)))
                explanation = getattr(result, "reasoning", None) or "No explanation provided"
                results.append((score, explanation))
            else:
                results.append((5.0, "Score extraction failed"))
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed for item {i}: {e}")
            results.append((None, str(e)))
    return results


def compute_rewards_batched(
    responses: List[str],
    batch: List[Dict],
    reward_model
) -> Tuple[List[float], List[float], List[float], List[str]]:
    """Compute rewards based on question hardness/difficulty."""
    judge_scores = []
    judge_rewards = []
    judge_explanations = []
    
    # Build all evaluation data - each response gets its OWN prompt/principle
    evaluations = []
    
    for i, (response, ex) in enumerate(zip(responses, batch)):
        # Use the original prompt that was given to the question generator
        prompt = ex.get("query", "")
        
        if not prompt:
            logger.warning(f"No prompt found in batch item {i}, skipping evaluation")
            judge_scores.append(0.0)
            judge_rewards.append(-1.0)
            judge_explanations.append("No prompt found in batch")
            continue
        
        # The principle contains the evaluation/scoring instructions
        principle = (
            "Score questions based on their DIFFICULTY and COMPLEXITY on a 0-10 scale. "
            "Rank higher (8-10) questions that require: multiple physics concepts, several calculation steps, "
            "complex reasoning, integration of different principles, or realistic multi-step scenarios. "
            "Rank lower (0-4) questions that are trivial, require only direct substitution, "
            "or involve a single simple formula. "
            "Focus on cognitive difficulty and problem-solving complexity, not on value accuracy or formatting. "
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
                # Use the original prompt that was given to the question generator
                prompt = ex.get("query", "")
                
                if not prompt:
                    logger.warning(f"No prompt found in batch item {i} during fallback")
                    judge_scores.append(0.0)
                    judge_rewards.append(-1.0)
                    judge_explanations.append("No prompt found in batch")
                    continue
                
                # The principle contains the evaluation/scoring instructions
                principle = (
                    "Score questions based on their DIFFICULTY and COMPLEXITY on a 0-10 scale. "
                    "Rank higher (8-10) questions that require: multiple physics concepts, several calculation steps, "
                    "complex reasoning, integration of different principles, or realistic multi-step scenarios. "
                    "Rank lower (0-4) questions that are trivial, require only direct substitution, "
                    "or involve a single simple formula. "
                    "Return scores as numeric values between 0 and 10."
                    "If the question is not concise penalize it."

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
    
    # Use judge rewards directly (no combination)
    return judge_rewards, judge_scores, judge_rewards, judge_explanations

