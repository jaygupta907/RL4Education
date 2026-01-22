"""
Graph data management utilities.
"""
import json
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def preload_graph_data(graph_file: str, min_tree_walk_length: int):
    """
    Pre-load graph data to avoid repeated file I/O.
    
    Returns:
        Tuple of (graph_data, deep_variables_cache, all_variables_cache)
    """
    logger.info("Pre-loading graph data...")
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
    
    # Pre-compute deep variables cache
    deep_variables_cache = get_variables_by_depth(graph_data, min_depth=min_tree_walk_length)
    all_variables_cache = [v['variable'] for v in graph_data['variables']]
    
    if not deep_variables_cache:
        logger.warning(f"No variables with depth >= {min_tree_walk_length} found. Using all variables.")
        deep_variables_cache = all_variables_cache.copy()
    else:
        logger.info(f"Found {len(deep_variables_cache)} variables with depth >= {min_tree_walk_length}")
    
    return graph_data, deep_variables_cache, all_variables_cache


def _get_all_dependencies(variable: str, graph_data: Dict) -> Set[str]:
    """
    Get all dependencies for a variable across all its formulas.
    Similar to TreeWalkCalculator._get_dependencies.
    """
    variable_info_map = {v['variable']: v for v in graph_data['variables']}
    if variable not in variable_info_map:
        return set()
    
    deps = set()
    var_info = variable_info_map[variable]
    for formula_entry in var_info.get('formulas', []):
        if isinstance(formula_entry, dict):
            deps.update(formula_entry.get('dependencies', []))
    
    return deps


def estimate_max_depth(variable: str, graph_data: Dict, visited: Set[str] = None, max_depth_limit: int = 10) -> int:
    """Estimate the maximum depth of dependency chain for a variable."""
    if visited is None:
        visited = set()
    
    if variable in visited or len(visited) >= max_depth_limit:
        return 0
    
    variable_info_map = {v['variable']: v for v in graph_data['variables']}
    if variable not in variable_info_map:
        return 0
    
    visited.add(variable)
    max_child_depth = 0
    
    # Get all dependencies from all formulas (not from top-level dependencies field)
    dependencies = _get_all_dependencies(variable, graph_data)
    
    for dep in dependencies:
        if dep not in visited:
            child_depth = estimate_max_depth(dep, graph_data, visited.copy(), max_depth_limit)
            max_child_depth = max(max_child_depth, child_depth)
    
    visited.remove(variable)
    return 1 + max_child_depth


def get_variables_by_depth(graph_data: Dict, min_depth: int = 3) -> List[str]:
    """Get variables that have at least min_depth levels of dependencies."""
    variable_info_map = {v['variable']: v for v in graph_data['variables']}
    deep_variables = []
    
    for var in variable_info_map.keys():
        depth = estimate_max_depth(var, graph_data)
        if depth >= min_depth:
            deep_variables.append(var)
    
    return deep_variables

