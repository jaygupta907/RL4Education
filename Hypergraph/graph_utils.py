"""
Hypergraph data management utilities.
"""
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def preload_hypergraph_data(hypergraph_file: str):
    """
    Pre-load hypergraph data to avoid repeated file I/O.
    
    Returns:
        Tuple of (hypergraph_data, all_nodes_cache)
    """
    logger.info("Pre-loading hypergraph data...")
    with open(hypergraph_file, 'r') as f:
        hypergraph_data = json.load(f)
    
    # Get all nodes
    all_nodes_cache = hypergraph_data.get('nodes', [])
    
    logger.info(f"Loaded {len(all_nodes_cache)} nodes from hypergraph")
    
    return hypergraph_data, all_nodes_cache

