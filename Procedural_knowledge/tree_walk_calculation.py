"""
Tree Walk Calculation Script

This script performs a tree walk from a target node in a variable dependency graph,
identifies leaf nodes, assigns random values to them, and then calculates backwards
to determine the target node's value.

Process:
1. Select a target node
2. Perform a tree walk backwards from target node to dependencies (max_length branches)
3. Identify leaf nodes (base inputs or nodes with no further dependencies)
4. Assign random values to leaf nodes
5. Calculate backwards from leaf nodes to target node, logging each step
6. Output the final calculated value for the target node

All steps are logged to console only.
"""

import json
import random
import math
import logging
from collections import deque
from typing import Dict, Set, List, Optional, Tuple

# Configure logging (console only, no file logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()  # Only console, no file logging
    ]
)
logger = logging.getLogger(__name__)


class TreeWalkCalculator:
    def __init__(self, graph_file: str, max_length: int = 3):
        """
        Initialize the TreeWalkCalculator.
        
        Args:
            graph_file: Path to the JSON file containing variable dependency graph
            max_length: Maximum length of each branch in the tree walk
        """
        self.max_length = max_length
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        self.variable_info = {v['variable']: v for v in graph_data['variables']}
        self.defined_variables = set(self.variable_info.keys())
        self.tree_structure = {}  # Stores the tree walk structure
        self.values = {}  # Stores calculated/assigned values
        # Create a mapping of variable to SI unit
        self.si_units = {v['variable']: v.get('si_unit', '') for v in graph_data['variables']}
    
    def _get_si_unit(self, variable: str) -> str:
        """Get SI unit for a variable."""
        return self.si_units.get(variable, '')
    
    def _get_dependencies(self, variable: str) -> List[str]:
        """
        Get combined dependencies for a variable across all its formulas.
        Uses explicit dependency lists from the JSON when available.
        """
        if variable not in self.variable_info:
            return []
        
        deps: Set[str] = set()
        for formula_entry in self.variable_info[variable].get('formulas', []):
            if isinstance(formula_entry, dict):
                deps.update(formula_entry.get('dependencies', []))
            else:
                deps.update(self._get_formula_dependencies(str(formula_entry)))
        return list(deps)
    
    def _select_random_formula(self, node: str) -> Optional[Tuple[str, Set[str]]]:
        """
        Randomly select a formula for a node and return (formula_string, dependencies).
        Only considers formulas with both a non-empty formula string and dependency list.
        """
        if node not in self.variable_info:
            return None
        
        valid_formulas: List[Tuple[str, Set[str]]] = []
        for formula_entry in self.variable_info[node].get('formulas', []):
            if isinstance(formula_entry, dict):
                formula_str = str(formula_entry.get('formula', '')).strip()
                deps = set(formula_entry.get('dependencies', []))
            else:
                formula_str = str(formula_entry).strip()
                deps = self._get_formula_dependencies(formula_str)
            
            if formula_str and deps:
                valid_formulas.append((formula_str, deps))
        
        if not valid_formulas:
            return None
        
        return random.choice(valid_formulas)
    
    def tree_walk(self, target_node: str) -> Dict:
        """
        Perform a tree walk from target node backwards to dependencies.
        At each node, selects ONE formula and visits ONLY the dependencies required by that formula.
        This ensures the tree only contains necessary nodes - no pruning needed.
        
        Algorithm:
        1. Start with target node at level 0
        2. For each node, select one formula from available formulas
        3. Visit only the dependencies used in that selected formula
        4. Prevent cycles by checking if a node was already visited
        5. Continue until max_length is reached or all dependencies are leaf nodes
        
        The walk continues to a fixed length (max_length) unless stopped by:
        - Leaf nodes: nodes with no dependencies, no valid formulas, or base inputs
        - Cycles: nodes already visited in the tree (skipped to prevent cycles)
        
        Returns the tree structure with selected formulas. Only necessary nodes are included.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting tree walk from target node: {target_node}")
        logger.info(f"Maximum branch length: {self.max_length}")
        logger.info(f"{'='*60}\n")
        
        tree = {
            'target': target_node,
            'nodes': set(),
            'edges': [],
            'leaf_nodes': set(),
            'levels': {},  # Store nodes by level
            'base_inputs': set(),  # Store base input nodes (not in defined_variables)
            'node_formulas': {}  # Store which formula is used for each node: node -> (formula, required_deps)
        }
        
        # Track the level at which each node was first encountered to prevent cycles
        node_levels = {}  # node -> level where it was first added
        visited_nodes = set()  # Track all nodes visited so far
        
        # BFS from target node going backwards
        queue = deque([(target_node, 0)])  # (node, level)
        tree['nodes'].add(target_node)
        tree['levels'][0] = [target_node]
        node_levels[target_node] = 0
        visited_nodes.add(target_node)
        
        while queue:
            current_node, level = queue.popleft()
            
            # Check if we've reached max length - mark as leaf and stop this branch
            if level >= self.max_length:
                tree['leaf_nodes'].add(current_node)
                logger.info(f"  Node {current_node} at level {level} marked as leaf (max length {self.max_length} reached)")
                continue
            
            # If current_node is not a defined variable, it's a base input (leaf) - stop this branch
            if current_node not in self.defined_variables:
                tree['leaf_nodes'].add(current_node)
                tree['base_inputs'].add(current_node)
                logger.info(f"  Found base input leaf node: {current_node}")
                continue
            
            # Get all possible dependencies for this node
            all_dependencies = set(self._get_dependencies(current_node))
            
            # Check if this variable has any valid formulas
            # Variables with empty formulas/dependencies should be treated as leaf nodes
            has_valid_formulas = False
            if current_node in self.variable_info:
                formulas = self.variable_info[current_node].get('formulas', [])
                # Check for at least one formula with a formula string and dependencies
                for f in formulas:
                    if isinstance(f, dict):
                        if f.get('formula') and f.get('dependencies'):
                            has_valid_formulas = True
                            break
                    elif isinstance(f, str) and f.strip():
                        has_valid_formulas = True
                        break
            
            # If no dependencies OR no valid formulas, this is a leaf node - stop this branch
            if not all_dependencies or not has_valid_formulas:
                tree['leaf_nodes'].add(current_node)
                if not all_dependencies:
                    logger.info(f"  Found leaf node at level {level}: {current_node} (no dependencies)")
                else:
                    logger.info(f"  Found leaf node at level {level}: {current_node} (no valid formulas)")
                continue
            
            # Separate dependencies into defined variables and base inputs
            defined_deps = {dep for dep in all_dependencies if dep in self.defined_variables}
            base_input_deps = {dep for dep in all_dependencies if dep not in self.defined_variables}
            
            # Randomly choose a formula for the current node
            formula_result = self._select_random_formula(current_node)
            
            if formula_result:
                formula, required_deps = formula_result
                tree['node_formulas'][current_node] = (formula, required_deps)
                logger.info(f"  Node {current_node} at level {level} uses formula: {formula}")
                logger.info(f"    Required dependencies: {sorted(required_deps)}")
                
                # Only visit dependencies required by the chosen formula
                # Separate into base inputs (leaf nodes) and defined variables (may need calculation)
                required_defined_deps = required_deps & defined_deps
                required_base_inputs = required_deps & base_input_deps
                
                # Add required base inputs as leaf nodes (they don't need to be visited)
                for base_input in required_base_inputs:
                    if base_input not in tree['nodes']:
                        tree['nodes'].add(base_input)
                        tree['base_inputs'].add(base_input)
                        tree['leaf_nodes'].add(base_input)
                    if (base_input, current_node) not in tree['edges']:
                        tree['edges'].append((base_input, current_node))
                    logger.info(f"    Added base input: {base_input}")
                
                # Visit only the required defined dependencies from the selected formula
                # These are the ONLY nodes we'll visit - no pruning needed later
                next_level = level + 1
                
                # Only continue to next level if we haven't reached max_length
                if next_level < self.max_length:
                    if next_level not in tree['levels']:
                        tree['levels'][next_level] = []
                    
                    for dep in required_defined_deps:
                        # Check if this dependency has valid formulas - if not, treat as leaf node
                        dep_has_valid_formulas = False
                        if dep in self.variable_info:
                            dep_formulas = self.variable_info[dep].get('formulas', [])
                            for f in dep_formulas:
                                if isinstance(f, dict):
                                    if f.get('formula') and f.get('dependencies'):
                                        dep_has_valid_formulas = True
                                        break
                                elif isinstance(f, str) and f.strip():
                                    dep_has_valid_formulas = True
                                    break
                        
                        # If dependency has no valid formulas, treat it as a leaf node instead of visiting it
                        if not dep_has_valid_formulas:
                            tree['leaf_nodes'].add(dep)
                            tree['nodes'].add(dep)
                            tree['edges'].append((dep, current_node))
                            logger.info(f"    Treating {dep} as leaf node (no valid formulas)")
                            continue
                        
                        # Prevent cycles: if node already exists in the tree at a different level, skip visiting it again
                        # But still add the edge to show the dependency relationship
                        if dep in node_levels:
                            existing_level = node_levels[dep]
                            if existing_level != next_level:
                                logger.info(f"    Skipping {dep} (already at level {existing_level}, would create cycle)")
                                # Still add the edge for visualization (shows dependency even if not recalculated)
                                if (dep, current_node) not in tree['edges']:
                                    tree['edges'].append((dep, current_node))
                                continue
                            # If it's at the same level, it's already queued, skip
                            continue
                        
                        # Add node to tree and queue for next level - this is the ONLY way nodes get added
                        tree['nodes'].add(dep)
                        tree['edges'].append((dep, current_node))
                        node_levels[dep] = next_level
                        visited_nodes.add(dep)
                        queue.append((dep, next_level))
                        tree['levels'][next_level].append(dep)
                        logger.info(f"    Visiting dependency: {dep} (level {next_level})")
                else:
                    # We've reached max_length, mark current node as leaf
                    tree['leaf_nodes'].add(current_node)
                    logger.info(f"  Node {current_node} marked as leaf (next level {next_level} would exceed max_length {self.max_length})")
            else:
                # No suitable formula found - mark as leaf node
                # This should not happen if formulas are properly defined, but handle gracefully
                logger.warning(f"  Node {current_node} at level {level}: No suitable formula found")
                logger.warning(f"    All dependencies: {sorted(all_dependencies)}")
                
                # Mark as leaf node since we can't calculate it
                tree['leaf_nodes'].add(current_node)
                # Don't add a formula entry if none was found
                # Don't visit any dependencies - this node becomes a leaf
        
        # Safety check: Mark nodes at max_length as leaves if they haven't been marked yet
        # (This should not happen since we check during the walk, but it's a safety measure)
        for level in tree['levels'].keys():
            if level >= self.max_length:
                for node in tree['levels'][level]:
                    if node not in tree['leaf_nodes']:
                        tree['leaf_nodes'].add(node)
                        logger.info(f"  Safety check: Marked {node} at level {level} as leaf (max_length {self.max_length})")
        
        logger.info(f"\nTree walk complete:")
        logger.info(f"  Total nodes: {len(tree['nodes'])}")
        logger.info(f"  Total edges: {len(tree['edges'])}")
        logger.info(f"  Leaf nodes: {len(tree['leaf_nodes'])}")
        logger.info(f"  Base inputs: {len(tree['base_inputs'])}")
        logger.info(f"  Levels: {len(tree['levels'])}")
        logger.info(f"  Nodes with selected formulas: {len([n for n in tree['node_formulas'] if tree['node_formulas'][n][0] is not None])}")
        
        logger.info(f"\nNodes by level:")
        for level in sorted(tree['levels'].keys()):
            logger.info(f"  Level {level}: {tree['levels'][level]}")
        
        logger.info(f"\nLeaf nodes: {sorted(tree['leaf_nodes'])}")
        logger.info(f"Base inputs: {sorted(tree['base_inputs'])}")
        
        logger.info(f"\nSelected formulas:")
        for node in sorted(tree['node_formulas'].keys()):
            formula, deps = tree['node_formulas'][node]
            if formula:
                logger.info(f"  {node}: {formula}")
                logger.info(f"    Dependencies: {sorted(deps)}")
        
        self.tree_structure = tree
        return tree
    
    def _get_formula_dependencies(self, formula: str) -> Set[str]:
        """Extract variable names from a formula string."""
        import re
        if not formula:
            return set()
        # Keywords, functions, and constants to exclude
        excluded = {
            'math', 'abs', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'pi', 'e',
            'and', 'or', 'not', 'if', 'else', 'for', 'while', 'def', 'import', 'from', 
            'as', 'in', 'is', 'None', 'True', 'False'
        }
        
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
        return {id for id in identifiers if id not in excluded}
    
    def _find_working_formula(self, variable: str, available_deps: Set[str], exclude_deps: Set[str] = None) -> Optional[Tuple[str, Set[str]]]:
        """Find the first formula that can work with available dependencies. Returns (formula, required_deps).
        
        Args:
            variable: Variable name
            available_deps: Set of available dependencies
            exclude_deps: Set of dependencies to exclude (e.g., to avoid circular dependencies)
        """
        if variable not in self.variable_info:
            return None
        
        exclude_deps = exclude_deps or set()
        
        for formula_entry in self.variable_info[variable]['formulas']:
            if isinstance(formula_entry, dict):
                formula = formula_entry.get('formula', '')
                required_deps = set(formula_entry.get('dependencies', []))
            else:
                formula = formula_entry
                required_deps = self._get_formula_dependencies(formula)
            if required_deps.issubset(available_deps) and not required_deps & exclude_deps:
                return (formula, required_deps)
        
        return None
    
    def _creates_cycle(self, node: str, required_deps: Set[str], target: str, necessary_nodes: Set[str]) -> bool:
        """Check if using required_deps for node would create a cycle."""
        if target in required_deps or node in required_deps:
            return True
        
        # Check for indirect cycles
        for dep in required_deps:
            if dep in necessary_nodes:
                dep_deps = {d for d, t in self.tree_structure['edges'] if t == dep}
                if node in dep_deps:
                    return True
        return False
    
    def prune_tree(self) -> Dict:
        """
        Prune the tree to keep only branches necessary to calculate the target.
        Since formulas are already selected during tree_walk, this method just ensures
        we only keep nodes that are actually required by the selected formulas.
        """
        logger.info(f"\n{'='*60}")
        logger.info("Pruning tree to keep only necessary branches")
        logger.info(f"{'='*60}\n")
        
        target = self.tree_structure['target']
        
        # Start with target and work backwards using selected formulas
        necessary_nodes = {target}
        necessary_edges = []
        node_formulas = self.tree_structure.get('node_formulas', {})
        
        # Build set of required nodes by following selected formulas backwards
        changed = True
        while changed:
            changed = False
            for node in list(necessary_nodes):
                if node in node_formulas:
                    formula, required_deps = node_formulas[node]
                    if formula:  # Only if a formula was selected
                        for dep in required_deps:
                            if dep not in necessary_nodes and dep in self.tree_structure['nodes']:
                                necessary_nodes.add(dep)
                                changed = True
                                # Add edge if it exists in original tree
                                if (dep, node) in self.tree_structure['edges']:
                                    necessary_edges.append((dep, node))
                    else:
                        # Fallback: no formula selected, use all dependencies from edges
                        node_deps = {dep for dep, target_node in self.tree_structure['edges'] if target_node == node}
                        for dep in node_deps:
                            if dep not in necessary_nodes:
                                necessary_nodes.add(dep)
                                changed = True
                                if (dep, node) not in necessary_edges:
                                    necessary_edges.append((dep, node))
        
        # Filter edges to only include those between kept nodes
        necessary_edges = [(dep, target_node) for dep, target_node in necessary_edges 
                          if dep in necessary_nodes and target_node in necessary_nodes]
        
        # Log removed nodes for debugging
        removed_nodes = self.tree_structure['nodes'] - necessary_nodes
        if removed_nodes:
            logger.info(f"  Removed unnecessary nodes: {sorted(removed_nodes)}")
        
        # Prune the tree structure
        pruned_tree = {
            'target': target,
            'nodes': necessary_nodes,
            'edges': necessary_edges,
            'leaf_nodes': set(),
            'base_inputs': self.tree_structure['base_inputs'].intersection(necessary_nodes),
            'levels': {},
            'node_formulas': {k: v for k, v in node_formulas.items() if k in necessary_nodes}
        }
        
        # Rebuild levels for pruned tree using BFS from target
        queue = deque([(target, 0)])
        visited = {target}
        pruned_tree['levels'][0] = [target]
        nodes_with_outgoing_edges = set()
        
        while queue:
            current_node, level = queue.popleft()
            next_level = level + 1
            
            # Get dependencies (children) of current node in pruned tree
            children = {dep for dep, target_node in pruned_tree['edges'] if target_node == current_node}
            
            if children:
                nodes_with_outgoing_edges.update(children)
                if next_level not in pruned_tree['levels']:
                    pruned_tree['levels'][next_level] = []
                
                for child in children:
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, next_level))
                        pruned_tree['levels'][next_level].append(child)
        
        # Leaf nodes: base inputs OR nodes with no outgoing edges (deepest nodes)
        pruned_tree['leaf_nodes'] = (
            pruned_tree['base_inputs'] | 
            {node for node in necessary_nodes 
             if node != target and node not in nodes_with_outgoing_edges}
        )
        
        logger.info(f"  Leaf nodes in pruned tree: {sorted(pruned_tree['leaf_nodes'])}")
        
        original_nodes = len(self.tree_structure['nodes'])
        original_edges = len(self.tree_structure['edges'])
        pruned_nodes = len(pruned_tree['nodes'])
        pruned_edges = len(pruned_tree['edges'])
        
        logger.info(f"\nPruning complete:")
        logger.info(f"  Original nodes: {original_nodes}")
        logger.info(f"  Pruned nodes: {pruned_nodes}")
        logger.info(f"  Original edges: {original_edges}")
        logger.info(f"  Pruned edges: {pruned_edges}")
        logger.info(f"  Removed {original_nodes - pruned_nodes} nodes")
        logger.info(f"  Removed {original_edges - pruned_edges} edges")
        
        logger.info(f"\nPruned nodes by level:")
        for level in sorted(pruned_tree['levels'].keys()):
            logger.info(f"  Level {level}: {pruned_tree['levels'][level]}")
        
        self.tree_structure = pruned_tree
        return pruned_tree
    
    def assign_random_values_to_leaves(self, min_val: float = 1.0, max_val: float = 100.0):
        """Assign random values to leaf nodes."""
        logger.info(f"\n{'='*60}")
        logger.info("Assigning random values to leaf nodes")
        logger.info(f"{'='*60}\n")
        
        for leaf in sorted(self.tree_structure['leaf_nodes'] & self.tree_structure['nodes']):
            value = round(random.uniform(min_val, max_val), 2)
            self.values[leaf] = value
            si_unit = self._get_si_unit(leaf)
            if si_unit:
                logger.info(f"  {leaf} = {value:.2f} {si_unit}")
            else:
                logger.info(f"  {leaf} = {value:.2f}")
    
    def _safe_eval(self, formula: str, context: Dict[str, float]) -> Optional[float]:
        """
        Safely evaluate a formula with given context.
        Returns None if evaluation fails or required variables are missing.
        """
        try:
            # Create a safe evaluation context
            safe_dict = {
                'math': math,
                'abs': abs,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'pi': math.pi,
                'e': math.e,
            }
            
            # Add context variables
            safe_dict.update(context)
            
            # Evaluate the formula
            result = eval(formula, {"__builtins__": {}}, safe_dict)
            return float(result)
        except (NameError, TypeError, ZeroDivisionError, ValueError) as e:
            return None
    
    def _try_formulas(self, variable: str, available_values: Dict[str, float]) -> Optional[float]:
        """Try to calculate variable using available formulas and values."""
        if variable not in self.variable_info:
            return None
        
        formulas = self.variable_info[variable]['formulas']
        
        for formula_entry in formulas:
            if isinstance(formula_entry, dict):
                formula = formula_entry.get('formula', '')
            else:
                formula = formula_entry
            
            result = self._safe_eval(formula, available_values)
            if result is not None:
                logger.info(f"    Formula: {formula}")
                logger.info(f"    Result: {result:.4f}")
                return result
        
        return None
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph. Returns list of cycles (each cycle is a list of nodes)."""
        def _get_all_dependencies(node: str) -> set:
            """Get all dependencies for a node, checking pruning info first."""
            if 'node_formulas' in self.tree_structure and node in self.tree_structure['node_formulas']:
                _, required_deps = self.tree_structure['node_formulas'][node]
                if required_deps:
                    return set(required_deps) & self.tree_structure['nodes']
            return set(self._get_dependencies(node)) & self.tree_structure['nodes']
        
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str):
            """DFS to detect cycles."""
            if node in rec_stack:
                # Found a cycle - extract it from path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            deps = _get_all_dependencies(node)
            for dep in deps:
                if dep not in self.tree_structure['leaf_nodes']:  # Only check non-leaf nodes
                    dfs(dep)
            
            path.pop()
            rec_stack.remove(node)
        
        # Check all non-leaf nodes for cycles
        for node in self.tree_structure['nodes']:
            if node not in self.tree_structure['leaf_nodes'] and node not in visited:
                dfs(node)
        
        return cycles
    
    def calculate_backwards(self) -> Optional[float]:
        """
        Calculate values backwards from leaf nodes to target node.
        Returns the calculated value of the target node.
        Processes levels from deepest to shallowest, ensuring all dependencies are available.
        Detects and breaks cycles by assigning random values to nodes in cycles.
        """
        logger.info(f"\n{'='*60}")
        logger.info("Calculating values backwards from leaf nodes to target")
        logger.info(f"{'='*60}\n")
        
        target = self.tree_structure['target']
        
        # Detect cycles before calculation
        cycles = self._detect_cycles()
        if cycles:
            logger.info(f"\nDetected {len(cycles)} cycle(s). Breaking cycles by assigning random values...")
            cycle_nodes_to_break = set()
            for cycle in cycles:
                # Remove duplicates and keep unique cycle
                unique_cycle = []
                seen = set()
                for node in cycle:
                    if node not in seen:
                        unique_cycle.append(node)
                        seen.add(node)
                
                if len(unique_cycle) > 1:  # Only break cycles with 2+ nodes
                    logger.info(f"  Cycle detected: {' -> '.join(unique_cycle)} -> {unique_cycle[0]}")
                    # Break cycle by assigning value to the first node (or node closest to target)
                    # Prefer breaking nodes that are not the target
                    break_node = unique_cycle[0]
                    for node in unique_cycle:
                        if node != target:
                            break_node = node
                            break
                    
                    if break_node not in self.values:
                        self.values[break_node] = round(random.uniform(1.0, 10.0), 2)
                        self.tree_structure['leaf_nodes'].add(break_node)
                        cycle_nodes_to_break.add(break_node)
                        logger.info(f"  Breaking cycle: assigned random value to {break_node} = {self.values[break_node]:.2f} (treating as leaf node)")
            
            if cycle_nodes_to_break:
                logger.info(f"  Total nodes converted to leaf nodes to break cycles: {len(cycle_nodes_to_break)}\n")
        
        # Helper function to get all dependencies for a node (from any level)
        def _get_all_dependencies(node: str) -> set:
            """Get all dependencies for a node, using the formula selected during tree walk."""
            if 'node_formulas' in self.tree_structure and node in self.tree_structure['node_formulas']:
                formula, required_deps = self.tree_structure['node_formulas'][node]
                if required_deps:
                    return set(required_deps) & self.tree_structure['nodes']
            return set(self._get_dependencies(node)) & self.tree_structure['nodes']
        
        # Helper function to get the selected formula for a node
        def _get_selected_formula(node: str) -> Optional[str]:
            """Get the formula selected for this node during tree walk."""
            if 'node_formulas' in self.tree_structure and node in self.tree_structure['node_formulas']:
                formula, _ = self.tree_structure['node_formulas'][node]
                return formula
            return None
        
        # Helper function to check if all dependencies are available
        def _all_dependencies_available(node: str) -> bool:
            """Check if all dependencies for a node are already calculated."""
            deps = _get_all_dependencies(node)
            return all(dep in self.values for dep in deps)
        
        # Helper function to calculate a single node
        def _calculate_node(node: str) -> bool:
            """Calculate value for a single node using the selected formula. Returns True if successful."""
            if node in self.tree_structure['leaf_nodes']:
                si_unit = self._get_si_unit(node)
                if si_unit:
                    logger.info(f"\n  {node} is a leaf node, already has value: {self.values[node]:.2f} {si_unit}")
                else:
                    logger.info(f"\n  {node} is a leaf node, already has value: {self.values[node]:.2f}")
                return True
            
            # Check if all dependencies are available
            deps = _get_all_dependencies(node)
            missing_deps = [d for d in deps if d not in self.values]
            if missing_deps:
                logger.warning(f"\n  Skipping {node} - missing dependencies: {missing_deps}")
                return False
            
            had_previous_value = node in self.values
            previous_value = self.values.get(node)
            
            logger.info(f"\n  Calculating: {node}")
            if had_previous_value:
                logger.info(f"    Previous value: {previous_value:.4f}")
            
            # Use the formula selected during tree walk
            selected_formula = _get_selected_formula(node)
            if selected_formula:
                logger.info(f"    Using selected formula: {selected_formula}")
                result = self._safe_eval(selected_formula, self.values)
                if result is not None:
                    result = round(result, 2)
                    self.values[node] = result
                    logger.info(f"    Result: {result:.2f}")
                    if had_previous_value:
                        logger.info(f"    Updated value: {result:.2f}")
                    return True
                else:
                    logger.warning(f"    Selected formula failed, trying other formulas...")
            
            # Fallback: try all formulas if selected formula doesn't work
            result = self._try_formulas(node, self.values)
            
            if result is not None:
                result = round(result, 2)
                self.values[node] = result
                if had_previous_value:
                    logger.info(f"    Updated value: {result:.2f}")
                return True
            else:
                logger.warning(f"    Could not calculate {node} - invalid formula")
                if not had_previous_value:
                    self.values[node] = round(random.uniform(1.0, 100.0), 2)
                    logger.info(f"    Assigned default random value: {self.values[node]:.2f}")
                return False
        
        # Process levels from deepest to shallowest, but ensure dependencies are ready
        # We iterate multiple times until all nodes are calculated or no progress is made
        all_levels = sorted(self.tree_structure['levels'].keys(), reverse=True)
        max_iterations = len(all_levels) * 2  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            progress_made = False
            
            for level in all_levels:
                logger.info(f"\n--- Processing Level {level} ---")
                
                # Get nodes at this level that need calculation (not leaf nodes)
                nodes_to_calculate = [node for node in self.tree_structure['levels'][level] 
                                     if node not in self.tree_structure['leaf_nodes']]
                
                if not nodes_to_calculate:
                    continue
                
                logger.info(f"  Nodes to calculate at level {level}: {nodes_to_calculate}")
                
                # Calculate nodes whose dependencies are all available
                for node in nodes_to_calculate:
                    if node in self.values:
                        continue  # Already calculated
                    
                    if _all_dependencies_available(node):
                        if _calculate_node(node):
                            progress_made = True
            
            # If no progress was made, check for remaining cycles and break them
            if not progress_made:
                if target in self.values:
                    break
                
                # Detect any remaining cycles and break them
                remaining_cycles = self._detect_cycles()
                if remaining_cycles:
                    logger.info(f"\nDetected {len(remaining_cycles)} remaining cycle(s) during calculation. Breaking...")
                    for cycle in remaining_cycles:
                        unique_cycle = []
                        seen = set()
                        for node in cycle:
                            if node not in seen and node not in self.values:
                                unique_cycle.append(node)
                                seen.add(node)
                        
                        if unique_cycle:
                            # Break cycle by assigning value to first non-target node
                            break_node = unique_cycle[0]
                            for node in unique_cycle:
                                if node != target:
                                    break_node = node
                                    break
                            
                            if break_node not in self.values:
                                self.values[break_node] = round(random.uniform(1.0, 100.0), 2)
                                self.tree_structure['leaf_nodes'].add(break_node)
                                si_unit = self._get_si_unit(break_node)
                                if si_unit:
                                    logger.info(f"  Breaking cycle: assigned random value to {break_node} = {self.values[break_node]:.2f} {si_unit} (treating as leaf node)")
                                else:
                                    logger.info(f"  Breaking cycle: assigned random value to {break_node} = {self.values[break_node]:.2f} (treating as leaf node)")
                                progress_made = True  # Try again after breaking cycle
                    
                    if progress_made:
                        continue  # Retry calculation after breaking cycles
                
                # Try to calculate remaining nodes anyway (might have missing formulas)
                for level in all_levels:
                    nodes_to_calculate = [node for node in self.tree_structure['levels'][level] 
                                         if node not in self.tree_structure['leaf_nodes']]
                    for node in nodes_to_calculate:
                        if node not in self.values:
                            _calculate_node(node)
                break
        
        # Return target value
        if target in self.values:
            logger.info(f"\n{'='*60}")
            logger.info(f"Final target value: {target} = {self.values[target]:.2f}")
            logger.info(f"{'='*60}\n")
            return self.values[target]
        else:
            logger.error(f"Could not calculate target node: {target}")
            return None
    
    def run(self, target_node: str, min_val: float = 1.0, max_val: float = 100.0) -> Optional[float]:
        """
        Run the complete process:
        1. Tree walk from target node (selects one formula per node, visits only dependencies from that formula)
        2. Assign random values to leaves
        3. Calculate backwards to target
        
        The tree walk is already selective - at each node, one formula is chosen and only
        its dependencies are visited. No pruning is needed.
        
        Returns the calculated target value.
        """
        # Step 1: Tree walk (already selective - only visits dependencies from selected formulas)
        self.tree_walk(target_node)
        
        # Step 2: Assign random values to leaves
        # No pruning needed - tree walk already only includes necessary nodes
        self.assign_random_values_to_leaves(min_val, max_val)
        
        # Step 3: Calculate backwards
        result = self.calculate_backwards()
        
        return result
    
    def print_summary(self):
        """Print a summary of all calculated values."""
        logger.info(f"\n{'='*60}")
        logger.info("Summary of all calculated values")
        logger.info(f"{'='*60}\n")
        
        # Print leaf nodes (these are the inputs used for question generation)
        leaf_nodes = self.tree_structure.get('leaf_nodes', set())
        if leaf_nodes:
            logger.info("Leaf Nodes (Input Values):")
            for node in sorted(leaf_nodes):
                if node in self.values:
                    si_unit = self._get_si_unit(node)
                    if si_unit:
                        logger.info(f"  {node} = {self.values[node]:.2f} {si_unit}")
                    else:
                        logger.info(f"  {node} = {self.values[node]:.2f}")
            logger.info("")
        
        # Print by level (excluding leaf nodes)
        for level in sorted(self.tree_structure['levels'].keys()):
            level_nodes = [node for node in sorted(self.tree_structure['levels'][level]) 
                          if node not in leaf_nodes and node in self.values]
            if level_nodes:
                logger.info(f"Level {level}:")
                for node in level_nodes:
                    logger.info(f"  {node} = {self.values[node]:.2f}")


def main():
    """Main function to run the tree walk calculation."""
    # Configuration
    graph_file = "variable_concept_graph.json"
    target_node = "magnetic_flux"
    max_length = 4
    min_val = 1.0
    max_val = 100.0
    
    # Create calculator
    calculator = TreeWalkCalculator(graph_file, max_length=max_length)
    
    # Run the calculation
    result = calculator.run(target_node, min_val=min_val, max_val=max_val)
    
    # Print summary
    calculator.print_summary()
    
    return result


if __name__ == "__main__":
    main()

