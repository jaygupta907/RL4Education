"""
Hypergraph trace generation utilities.
"""
import random
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from hypergraph_traverser import HypergraphTraverser
from prompt_generator import create_prompt as create_hypergraph_prompt

logger = logging.getLogger(__name__)


def generate_single_trace(
    hypergraph_file: str,
    max_depth: int,
    max_traces: int,
    min_trace_length: int,
    all_nodes_cache: List[str],
    tokenizer
) -> Optional[Dict]:
    """Generate a single hypergraph trace (for parallel processing)."""
    try:
        # Select target randomly
        target = random.choice(all_nodes_cache)
        
        # Create traverser
        traverser = HypergraphTraverser(hypergraph_file)
        
        # Find all traces for this target
        traces = traverser.get_all_traces_formatted(
            target=target,
            max_depth=max_depth,
            max_traces=max_traces
        )
        
        if not traces:
            return None
        
        # Select a random trace
        trace = random.choice(traces)
        
        # Check trace length
        trace_length = len(trace.get('formulas', []))
        
        if trace_length < min_trace_length:
            return None
        
        # Create prompt
        prompt, metadata = create_hypergraph_prompt(traverser, trace, target, tokenizer)
        
        return {
            "query": prompt,
            "traverser": traverser,
            "trace": trace,
            "target": target,
            "metadata": metadata,
        }
    except Exception as e:
        logger.debug(f"Error generating trace: {e}")
        return None


def collect_batch_parallel(
    batch_size: int,
    num_workers: int,
    hypergraph_file: str,
    max_depth: int,
    max_traces: int,
    min_trace_length: int,
    all_nodes_cache: List[str],
    tokenizer
) -> List[Dict]:
    """OPTIMIZATION: Collect batch using parallel trace generation."""
    batch = []
    max_attempts = batch_size * 50
    
    # Use ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit multiple tasks
        futures = []
        for _ in range(max_attempts):
            future = executor.submit(
                generate_single_trace,
                hypergraph_file,
                max_depth,
                max_traces,
                min_trace_length,
                all_nodes_cache,
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
                logger.debug(f"Trace generation failed: {e}")
    
    if len(batch) < batch_size:
        logger.warning(f"Only collected {len(batch)}/{batch_size} valid traces")
    
    return batch[:batch_size]

