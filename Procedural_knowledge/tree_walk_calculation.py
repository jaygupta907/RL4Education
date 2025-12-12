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

All steps are logged to both console and tree_walk_calculation.log file.
"""

import json
import random
import math
import logging
from collections import deque
from typing import Dict, Set, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('tree_walk_calculation.log'),
        logging.StreamHandler()
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
        self.graph_data = self._load_graph(graph_file)
        self.variable_info = {v['variable']: v for v in self.graph_data['variables']}
        self.defined_variables = set(self.variable_info.keys())
        self.tree_structure = {}  # Stores the tree walk structure
        self.values = {}  # Stores calculated/assigned values
        
    def _load_graph(self, graph_file: str) -> dict:
        """Load the graph from JSON file."""
        with open(graph_file, 'r') as f:
            return json.load(f)
    
    def _get_dependencies(self, variable: str) -> List[str]:
        """Get dependencies for a variable."""
        if variable in self.variable_info:
            return self.variable_info[variable]['dependencies']
        return []
    
    def _is_leaf_node(self, variable: str) -> bool:
        """Check if a variable is a leaf node (no dependencies or dependencies not in graph)."""
        deps = self._get_dependencies(variable)
        if not deps:
            return True
        # Check if all dependencies are not in the defined variables
        return all(dep not in self.defined_variables for dep in deps)
    
    def tree_walk(self, target_node: str) -> Dict:
        """
        Perform a tree walk from target node backwards to dependencies.
        Returns the tree structure.
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
            'base_inputs': set()  # Store base input nodes (not in defined_variables)
        }
        
        # Track the level at which each node was first encountered to prevent cycles
        node_levels = {}  # node -> level where it was first added
        
        # BFS from target node going backwards
        queue = deque([(target_node, 0)])  # (node, level)
        tree['nodes'].add(target_node)
        tree['levels'][0] = [target_node]
        node_levels[target_node] = 0
        
        while queue:
            current_node, level = queue.popleft()
            
            # Check if we've reached max length
            if level >= self.max_length:
                # Mark as leaf if we've reached max length
                tree['leaf_nodes'].add(current_node)
                continue
            
            # Get dependencies (going backwards in the dependency graph)
            dependencies = self._get_dependencies(current_node)
            
            # If no dependencies or all dependencies are base inputs, mark as leaf
            if not dependencies:
                tree['leaf_nodes'].add(current_node)
                logger.info(f"  Found leaf node at level {level}: {current_node} (no dependencies)")
                continue
            
            # Separate dependencies into defined variables and base inputs
            defined_deps = [dep for dep in dependencies if dep in self.defined_variables]
            base_input_deps = [dep for dep in dependencies if dep not in self.defined_variables]
            
            # Add base input dependencies as leaf nodes
            for base_input in base_input_deps:
                if base_input not in tree['nodes']:
                    tree['nodes'].add(base_input)
                    tree['base_inputs'].add(base_input)
                    tree['leaf_nodes'].add(base_input)
                    tree['edges'].append((base_input, current_node))
                    logger.info(f"  Found base input leaf node: {base_input}")
            
            # If no defined dependencies, current node is a leaf
            if not defined_deps:
                tree['leaf_nodes'].add(current_node)
                logger.info(f"  Found leaf node at level {level}: {current_node} (only base inputs)")
                continue
            
            # Add defined dependencies to tree and queue
            next_level = level + 1
            if next_level not in tree['levels']:
                tree['levels'][next_level] = []
            
            for dep in defined_deps:
                # Prevent cycles: if node already exists in the tree, skip it
                # This ensures each node appears only once, preventing circular dependencies
                if dep in node_levels:
                    logger.info(f"  Skipping {dep} (already at level {node_levels[dep]}, would create cycle)")
                    # Still add the edge for visualization, but don't process the node again
                    if (dep, current_node) not in tree['edges']:
                        tree['edges'].append((dep, current_node))
                    continue
                
                # Add node to tree
                tree['nodes'].add(dep)
                tree['edges'].append((dep, current_node))
                node_levels[dep] = next_level
                queue.append((dep, next_level))
                tree['levels'][next_level].append(dep)
        
        # Mark nodes at max_length as leaves if they haven't been marked yet
        for level in tree['levels'].keys():
            if level >= self.max_length:
                for node in tree['levels'][level]:
                    if node not in tree['leaf_nodes']:
                        tree['leaf_nodes'].add(node)
        
        logger.info(f"\nTree walk complete:")
        logger.info(f"  Total nodes: {len(tree['nodes'])}")
        logger.info(f"  Total edges: {len(tree['edges'])}")
        logger.info(f"  Leaf nodes: {len(tree['leaf_nodes'])}")
        logger.info(f"  Base inputs: {len(tree['base_inputs'])}")
        logger.info(f"  Levels: {len(tree['levels'])}")
        
        logger.info(f"\nNodes by level:")
        for level in sorted(tree['levels'].keys()):
            logger.info(f"  Level {level}: {tree['levels'][level]}")
        
        logger.info(f"\nLeaf nodes: {sorted(tree['leaf_nodes'])}")
        logger.info(f"Base inputs: {sorted(tree['base_inputs'])}")
        
        self.tree_structure = tree
        return tree
    
    def _get_formula_dependencies(self, formula: str) -> Set[str]:
        """Extract variable names from a formula string."""
        import re
        # Get all variable names from the formula (words that are not Python keywords or functions)
        # This is a simple approach - match identifiers that aren't math functions or constants
        python_keywords = {'math', 'abs', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'pi', 'e', 'and', 'or', 'not', 'if', 'else', 'for', 'while', 'def', 'import', 'from', 'as', 'in', 'is', 'None', 'True', 'False'}
        math_functions = {'math', 'abs', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10'}
        constants = {'pi', 'e'}
        
        # Find all identifiers
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
        # Filter out keywords, functions, and constants
        deps = {id for id in identifiers if id not in python_keywords and id not in math_functions and id not in constants}
        return deps
    
    def _find_working_formula(self, variable: str, available_deps: Set[str], exclude_deps: Set[str] = None) -> Optional[Tuple[str, Set[str]]]:
        """Find the first formula that can work with available dependencies. Returns (formula, required_deps).
        
        Args:
            variable: Variable name
            available_deps: Set of available dependencies
            exclude_deps: Set of dependencies to exclude (e.g., to avoid circular dependencies)
        """
        if variable not in self.variable_info:
            return None
        
        if exclude_deps is None:
            exclude_deps = set()
        
        formulas = self.variable_info[variable]['formulas']
        
        for formula in formulas:
            required_deps = self._get_formula_dependencies(formula)
            # Check if all required dependencies are available and not excluded
            if required_deps.issubset(available_deps) and not required_deps.intersection(exclude_deps):
                return (formula, required_deps)
        
        return None
    
    def prune_tree(self) -> Dict:
        """
        Prune the tree to keep only branches necessary to calculate the target.
        Works backwards from target, keeping only dependencies that are actually used by formulas.
        """
        logger.info(f"\n{'='*60}")
        logger.info("Pruning tree to keep only necessary branches")
        logger.info(f"{'='*60}\n")
        
        target = self.tree_structure['target']
        
        # Start with just the target, then recursively add only needed dependencies
        necessary_nodes = {target}
        necessary_edges = []
        node_formulas = {}  # Store which formula and dependencies are used for each node
        
        # Work backwards from target, level by level (from shallowest to deepest)
        all_levels = sorted(self.tree_structure['levels'].keys())
        
        # Process each level to determine which dependencies are needed
        # We need to process multiple passes to ensure we capture all dependencies
        changed = True
        while changed:
            changed = False
            for level in all_levels:
                for node in self.tree_structure['levels'][level]:
                    if node not in necessary_nodes:
                        continue
                    
                    # Get all dependencies of this node from the tree edges
                    node_deps = {dep for dep, target_node in self.tree_structure['edges'] if target_node == node}
                    
                    # Also get all possible dependencies from variable definition (some might not have edges)
                    if node in self.variable_info:
                        all_possible_deps = set(self.variable_info[node]['dependencies'])
                    else:
                        all_possible_deps = node_deps
                    
                    if not node_deps and not all_possible_deps:
                        continue
                    
                    # Find which formula works and what dependencies it needs
                    # Check against what's available in necessary_nodes (pruned tree)
                    # Use all_possible_deps to find formulas, but only add dependencies that are in node_deps or necessary_nodes
                    available_deps = all_possible_deps.intersection(necessary_nodes)
                    
                    if node in self.defined_variables:
                        result = self._find_working_formula(node, available_deps)
                        if result:
                            formula, required_deps = result
                            # Check for circular dependency: don't use formula if it depends on the target, itself, or nodes that depend on it
                            # Check if any required dependency already has this node as a dependency (circular)
                            creates_cycle = False
                            if target in required_deps or node in required_deps:
                                creates_cycle = True
                            else:
                                # Check if any required dependency depends on this node (indirect cycle)
                                for dep in required_deps:
                                    if dep in necessary_nodes:
                                        dep_deps = {d for d, t in self.tree_structure['edges'] if t == dep}
                                        if node in dep_deps:
                                            creates_cycle = True
                                            break
                            
                            if not creates_cycle:
                                if node not in node_formulas:
                                    node_formulas[node] = (formula, required_deps)
                                    logger.info(f"  Node {node} uses formula: {formula}")
                                    logger.info(f"    Required dependencies: {sorted(required_deps)}")
                                
                                # Add only required dependencies to necessary nodes
                                for dep in required_deps:
                                    if dep not in necessary_nodes:
                                        necessary_nodes.add(dep)
                                        changed = True
                                    # Add edge if it exists in original tree
                                    if (dep, node) not in necessary_edges and (dep, node) in self.tree_structure['edges']:
                                        necessary_edges.append((dep, node))
                            else:
                                # Circular dependency detected, skip this formula
                                result = None
                        
                        if not result:
                            # If no formula works with available deps, try checking all possible deps
                            # (some dependencies might be added in this iteration)
                            # First, find dependencies that would create cycles
                            cycle_deps = set()
                            if node in necessary_nodes:
                                node_deps_from_edges = {d for d, t in self.tree_structure['edges'] if t == node}
                                for dep in all_possible_deps:
                                    if dep in necessary_nodes:
                                        dep_deps = {d for d, t in self.tree_structure['edges'] if t == dep}
                                        if node in dep_deps:
                                            cycle_deps.add(dep)
                            
                            # Try finding formula excluding cycle dependencies
                            result_all = self._find_working_formula(node, all_possible_deps, exclude_deps=cycle_deps)
                            if result_all:
                                formula, required_deps = result_all
                                # Double-check for circular dependency
                                creates_cycle = False
                                if target in required_deps or node in required_deps:
                                    creates_cycle = True
                                else:
                                    for dep in required_deps:
                                        if dep in necessary_nodes:
                                            dep_deps = {d for d, t in self.tree_structure['edges'] if t == dep}
                                            if node in dep_deps:
                                                creates_cycle = True
                                                break
                                
                                if not creates_cycle:
                                    # Only use this formula if all required deps are in node_deps (will be added)
                                    if node not in node_formulas:
                                        node_formulas[node] = (formula, required_deps)
                                        logger.info(f"  Node {node} uses formula: {formula}")
                                        logger.info(f"    Required dependencies: {sorted(required_deps)}")
                                    
                                    for dep in required_deps:
                                        if dep not in necessary_nodes:
                                            necessary_nodes.add(dep)
                                            changed = True
                                        if (dep, node) not in necessary_edges and (dep, node) in self.tree_structure['edges']:
                                            necessary_edges.append((dep, node))
                                else:
                                    # Circular dependency, try next formula or fallback
                                    result_all = None
                            
                            if not result_all:
                                # If no formula works, keep all dependencies (fallback)
                                if node not in node_formulas:
                                    logger.warning(f"  Node {node}: No formula found with available dependencies, keeping all")
                                    node_formulas[node] = (None, node_deps)
                                for dep in node_deps:
                                    if dep not in necessary_nodes:
                                        necessary_nodes.add(dep)
                                        changed = True
                                    if (dep, node) not in necessary_edges and (dep, node) in self.tree_structure['edges']:
                                        necessary_edges.append((dep, node))
        
        # Cleanup: Remove nodes that aren't actually used by selected formulas
        # Build set of all nodes that are required dependencies of selected formulas
        required_nodes = {target}
        changed = True
        while changed:
            changed = False
            # For each node that's already required, add its required dependencies
            for node in list(required_nodes):
                if node in node_formulas:
                    formula, required_deps = node_formulas[node]
                    if formula:  # Only if a formula was selected
                        for dep in required_deps:
                            if dep not in required_nodes:
                                required_nodes.add(dep)
                                changed = True
                    else:
                        # Fallback case: no formula selected, but dependencies were kept
                        # Use the dependencies that were stored (node_deps)
                        for dep in required_deps:
                            if dep not in required_nodes and dep in necessary_nodes:
                                required_nodes.add(dep)
                                changed = True
        
        # Only keep nodes that are actually required by selected formulas
        # Base inputs are only kept if they're actually required (in required_nodes)
        nodes_before_cleanup = necessary_nodes.copy()
        necessary_nodes = necessary_nodes.intersection(required_nodes)
        
        # Also filter edges to only include those between kept nodes
        necessary_edges = [(dep, target_node) for dep, target_node in necessary_edges 
                          if dep in necessary_nodes and target_node in necessary_nodes]
        
        # Log removed nodes for debugging
        removed_nodes = nodes_before_cleanup - necessary_nodes
        if removed_nodes:
            logger.info(f"  Cleanup removed unnecessary nodes: {sorted(removed_nodes)}")
        
        # Prune the tree structure
        pruned_tree = {
            'target': target,
            'nodes': necessary_nodes,
            'edges': necessary_edges,
            'leaf_nodes': set(),
            'base_inputs': self.tree_structure['base_inputs'].intersection(necessary_nodes),
            'levels': {},
            'node_formulas': node_formulas  # Store which formula is used for each node
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
        
        # Leaf nodes in pruned tree are nodes that:
        # 1. Are base inputs that are in the pruned tree, OR
        # 2. Have no outgoing edges (no nodes in pruned tree depend on them) AND are in necessary_nodes
        pruned_tree['leaf_nodes'] = set()
        # Add base inputs that are in the pruned tree
        for node in pruned_tree['base_inputs']:
            if node in necessary_nodes:
                pruned_tree['leaf_nodes'].add(node)
        # Add nodes with no outgoing edges (these are the deepest nodes in the pruned tree)
        for node in necessary_nodes:
            if node not in nodes_with_outgoing_edges and node != target:
                pruned_tree['leaf_nodes'].add(node)
        
        logger.info(f"  Leaf nodes in pruned tree: {sorted(pruned_tree['leaf_nodes'])}")
        
        logger.info(f"\nPruning complete:")
        logger.info(f"  Original nodes: {len(self.tree_structure['nodes'])}")
        logger.info(f"  Pruned nodes: {len(pruned_tree['nodes'])}")
        logger.info(f"  Original edges: {len(self.tree_structure['edges'])}")
        logger.info(f"  Pruned edges: {len(pruned_tree['edges'])}")
        logger.info(f"  Removed {len(self.tree_structure['nodes']) - len(pruned_tree['nodes'])} nodes")
        logger.info(f"  Removed {len(self.tree_structure['edges']) - len(pruned_tree['edges'])} edges")
        
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
        
        # Only assign to leaf nodes that are actually in the pruned tree
        leaf_nodes = self.tree_structure['leaf_nodes'].intersection(self.tree_structure['nodes'])
        
        for leaf in sorted(leaf_nodes):
            value = random.uniform(min_val, max_val)
            self.values[leaf] = value
            logger.info(f"  {leaf} = {value:.4f}")
    
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
        
        for formula in formulas:
            result = self._safe_eval(formula, available_values)
            if result is not None:
                logger.info(f"    Formula: {formula}")
                logger.info(f"    Result: {result:.4f}")
                return result
        
        return None
    
    def calculate_backwards(self) -> Optional[float]:
        """
        Calculate values backwards from leaf nodes to target node.
        Returns the calculated value of the target node.
        Note: Skips the deepest level (leaf nodes) since they already have assigned values.
        """
        logger.info(f"\n{'='*60}")
        logger.info("Calculating values backwards from leaf nodes to target")
        logger.info(f"{'='*60}\n")
        
        target = self.tree_structure['target']
        
        # Process nodes level by level, from deepest to shallowest (highest level number to lowest)
        # This ensures dependencies are calculated before nodes that depend on them
        all_levels = sorted(self.tree_structure['levels'].keys(), reverse=True)
        
        # Process all levels, but skip leaf nodes (they already have assigned values)
        # The deepest level may contain both leaf nodes (with values) and non-leaf nodes (to calculate)
        levels_to_process = all_levels
        
        for level in levels_to_process:
            logger.info(f"\n--- Processing Level {level} ---")
            
            # Get nodes at this level that need calculation (not leaf nodes)
            nodes_to_calculate = [node for node in self.tree_structure['levels'][level] 
                                 if node not in self.tree_structure['leaf_nodes']]
            
            # Process nodes in topological order within the level
            # Nodes that don't depend on other nodes in the same level should be processed first
            processed = set()
            level_nodes_set = set(nodes_to_calculate)
            
            # Build dependency graph for this level
            # Use the formula selected during pruning to determine actual dependencies
            level_dependencies = {}
            logger.info(f"  Nodes to calculate at level {level}: {nodes_to_calculate}")
            for node in nodes_to_calculate:
                # Check if we have a formula selected during pruning
                if 'node_formulas' in self.tree_structure and node in self.tree_structure['node_formulas']:
                    formula, required_deps = self.tree_structure['node_formulas'][node]
                    if formula:  # If a formula was selected
                        # Only consider dependencies that are actually needed for this formula
                        same_level_deps = set(required_deps).intersection(level_nodes_set).intersection(self.tree_structure['nodes'])
                    else:
                        # No formula found during pruning, use all dependencies
                        all_deps = set(self._get_dependencies(node))
                        same_level_deps = all_deps.intersection(level_nodes_set).intersection(self.tree_structure['nodes'])
                else:
                    # No pruning info, use all dependencies
                    all_deps = set(self._get_dependencies(node))
                    same_level_deps = all_deps.intersection(level_nodes_set).intersection(self.tree_structure['nodes'])
                
                level_dependencies[node] = same_level_deps
                if same_level_deps:
                    logger.info(f"  {node} depends on same-level nodes: {same_level_deps}")
            
            # Keep processing until all nodes are done
            while len(processed) < len(nodes_to_calculate):
                progress_made = False
                
                for node in nodes_to_calculate:
                    if node in processed:
                        continue
                    
                    # Check if all dependencies in the same level are already processed
                    same_level_deps = level_dependencies.get(node, set())
                    
                    if same_level_deps.issubset(processed):
                        # All same-level dependencies are processed, we can calculate this node
                        processed.add(node)
                        progress_made = True
                        
                        # Skip if it's a leaf node (shouldn't happen, but safety check)
                        if node in self.tree_structure['leaf_nodes']:
                            logger.info(f"\n  {node} is a leaf node, already has value: {self.values[node]:.4f}")
                            continue
                        
                        # If node already has a value from a previous calculation, try to recalculate
                        # with newly available dependencies
                        had_previous_value = node in self.values
                        previous_value = self.values.get(node)
                        
                        logger.info(f"\n  Calculating: {node}")
                        if had_previous_value:
                            logger.info(f"    Previous value: {previous_value:.4f}")
                        
                        # Try to calculate using formulas
                        result = self._try_formulas(node, self.values)
                        
                        if result is not None:
                            self.values[node] = result
                            if had_previous_value:
                                logger.info(f"    Updated value: {result:.4f}")
                        else:
                            logger.warning(f"    Could not calculate {node} - missing dependencies or invalid formula")
                            # Check which dependencies are missing
                            deps = self._get_dependencies(node)
                            missing = [d for d in deps if d not in self.values]
                            if missing:
                                logger.warning(f"    Missing dependencies: {missing}")
                            # Only assign default if we don't have a previous value
                            if not had_previous_value:
                                self.values[node] = random.uniform(1.0, 100.0)
                                logger.info(f"    Assigned default random value: {self.values[node]:.4f}")
                
                if not progress_made:
                    # No progress made, process remaining nodes anyway (circular dependencies or missing deps)
                    for node in nodes_to_calculate:
                        if node not in processed:
                            processed.add(node)
                            
                            if node in self.tree_structure['leaf_nodes']:
                                logger.info(f"\n  {node} is a leaf node, already has value: {self.values[node]:.4f}")
                                continue
                            
                            had_previous_value = node in self.values
                            previous_value = self.values.get(node)
                            
                            logger.info(f"\n  Calculating: {node}")
                            if had_previous_value:
                                logger.info(f"    Previous value: {previous_value:.4f}")
                            
                            result = self._try_formulas(node, self.values)
                            
                            if result is not None:
                                self.values[node] = result
                                if had_previous_value:
                                    logger.info(f"    Updated value: {result:.4f}")
                            else:
                                logger.warning(f"    Could not calculate {node} - missing dependencies or invalid formula")
                                deps = self._get_dependencies(node)
                                missing = [d for d in deps if d not in self.values]
                                if missing:
                                    logger.warning(f"    Missing dependencies: {missing}")
                                if not had_previous_value:
                                    self.values[node] = random.uniform(1.0, 100.0)
                                    logger.info(f"    Assigned default random value: {self.values[node]:.4f}")
                    break
        
        # Return target value
        if target in self.values:
            logger.info(f"\n{'='*60}")
            logger.info(f"Final target value: {target} = {self.values[target]:.4f}")
            logger.info(f"{'='*60}\n")
            return self.values[target]
        else:
            logger.error(f"Could not calculate target node: {target}")
            return None
    
    def run(self, target_node: str, min_val: float = 1.0, max_val: float = 100.0) -> Optional[float]:
        """
        Run the complete process:
        1. Tree walk from target node
        2. Prune tree to keep only necessary branches
        3. Assign random values to leaves
        4. Calculate backwards to target
        
        Returns the calculated target value.
        """
        # Step 1: Tree walk
        self.tree_walk(target_node)
        
        # Step 2: Prune tree to keep only necessary branches
        self.prune_tree()
        
        # Step 3: Assign random values to leaves
        self.assign_random_values_to_leaves(min_val, max_val)
        
        # Step 4: Calculate backwards
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
                    logger.info(f"  {node} = {self.values[node]:.4f}")
            logger.info("")
        
        # Print by level (excluding leaf nodes as they're already printed above)
        for level in sorted(self.tree_structure['levels'].keys()):
            level_nodes = [node for node in sorted(self.tree_structure['levels'][level]) 
                          if node not in leaf_nodes and node in self.values]
            if level_nodes:
                logger.info(f"Level {level}:")
                for node in level_nodes:
                    logger.info(f"  {node} = {self.values[node]:.4f}")


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

