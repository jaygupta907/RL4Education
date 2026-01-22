"""
Tree walk generation utilities.
"""
import random
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tree_walk_calculation import TreeWalkCalculator
from prompt_generator import create_prompt

logger = logging.getLogger(__name__)


def generate_single_tree_walk(
    graph_file: str,
    max_length: int,
    min_tree_walk_length: int,
    deep_variables_cache: List[str],
    all_variables_cache: List[str],
    tokenizer
) -> Optional[Dict]:
    """Generate a single tree walk (for parallel processing)."""
    try:
        # Select target
        if deep_variables_cache and random.random() < 0.9:
            target = random.choice(deep_variables_cache)
        else:
            target = random.choice(all_variables_cache)
        
        # Create calculator
        calculator = TreeWalkCalculator(
            graph_file,
            max_length=max_length
        )
        
        result = calculator.run(
            target,
            min_val=1.0,
            max_val=10.0
        )
        
        if result is None:
            return None
        
        # Check tree walk length
        tree_walk_length = 0
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
        
        if tree_walk_length < min_tree_walk_length:
            return None
        
        # Create prompt
        prompt, metadata = create_prompt(calculator, tokenizer)
        
        return {
            "query": prompt,
            "calculator": calculator,
            "metadata": metadata,
        }
    except Exception as e:
        logger.debug(f"Error generating tree walk: {e}")
        return None


def collect_batch_parallel(
    batch_size: int,
    num_workers: int,
    graph_file: str,
    max_length: int,
    min_tree_walk_length: int,
    deep_variables_cache: List[str],
    all_variables_cache: List[str],
    tokenizer
) -> List[Dict]:
    """OPTIMIZATION: Collect batch using parallel tree walk generation."""
    batch = []
    max_attempts = batch_size * 50
    
    # Use ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit multiple tasks
        futures = []
        for _ in range(max_attempts):
            future = executor.submit(
                generate_single_tree_walk,
                graph_file,
                max_length,
                min_tree_walk_length,
                deep_variables_cache,
                all_variables_cache,
                tokenizer
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            if len(batch) >= batch_size:
                break
            try:
                result = future.result(timeout=30)
                if result:
                    batch.append(result)
            except Exception as e:
                logger.debug(f"Tree walk generation failed: {e}")
    
    if len(batch) < batch_size:
        logger.warning(f"Only collected {len(batch)}/{batch_size} valid tree walks")
    
    return batch[:batch_size]

