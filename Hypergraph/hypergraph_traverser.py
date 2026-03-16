"""
Hypergraph Traverser - Finds all solution traces for a target variable.

This module traverses the formula hypergraph backwards from a target node
to find all possible solution paths (traces) that can calculate the target.
Cycles are detected and variables causing cycles are treated as leaf nodes.
"""
import json
from typing import Dict, List, Set, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class HypergraphTraverser:
    """
    Traverses the hypergraph to find all solution traces for a target variable.
    """
    
    def __init__(self, hypergraph_file: str):
        """
        Initialize the traverser.
        
        Args:
            hypergraph_file: Path to formula_hypergraph.json
        """
        with open(hypergraph_file, 'r') as f:
            self.hypergraph = json.load(f)
        
        # Build index: variable -> list of hyperedges that produce it
        self.output_to_hyperedges: Dict[str, List[Dict]] = {}
        for hyperedge in self.hypergraph['hyperedges']:
            output = hyperedge['output']
            if output not in self.output_to_hyperedges:
                self.output_to_hyperedges[output] = []
            self.output_to_hyperedges[output].append(hyperedge)
        
        # Get all nodes for reference
        self.all_nodes = set(self.hypergraph['nodes'])

    # ------------------------------------------------------------------
    # Domain compatibility: which producer domains may feed into which
    # consumer domains.  A computed intermediate is only allowed as input
    # to a formula whose domain appears in the producer's allowed set.
    # Domains not listed here are self-contained (only feed themselves).
    # ------------------------------------------------------------------
    _DOMAIN_COMPAT: Dict[str, Set[str]] = {
        # kinematics outputs (velocity, displacement, acceleration) may feed
        # dynamics and energy — all physically natural chains
        "kinematics":      {"kinematics", "dynamics", "energy", "rotational", "waves"},
        # dynamics outputs (force, pressure, momentum) may feed thermo (gas laws
        # use mechanical pressure) but NOT geometry — F/A pressure must not
        # become a geometric area source
        "dynamics":        {"dynamics", "kinematics", "energy", "thermodynamics", "rotational"},
        "rotational":      {"rotational", "dynamics", "energy"},
        # energy outputs (KE, PE, height, work) stay within energy/kinematics —
        # height derived from PE must NOT feed geometry formulas (the key fix)
        "energy":          {"energy", "kinematics", "dynamics"},
        # thermodynamics is self-contained: density/volume/temperature stay thermo
        "thermodynamics":  {"thermodynamics"},
        # geometry outputs (area, volume) may feed thermo (e.g. V in ideal gas)
        # and elasticity (area in stress), but NOT energy or dynamics
        "geometry":        {"geometry", "thermodynamics", "elasticity"},
        "elasticity":      {"elasticity"},
        "friction":        {"friction", "dynamics"},
        "waves":           {"waves", "electromagnetism"},
        "electromagnetism":{"electromagnetism", "waves"},
        # unknown/untagged formulas are never blocked
        "unknown":         {"kinematics","dynamics","rotational","energy",
                            "thermodynamics","geometry","elasticity",
                            "friction","waves","electromagnetism","unknown"},
    }

    def _check_domain_coherence(self, execution_order: List[Dict]) -> bool:
        """
        Reject traces where an intermediate variable computed in domain A
        is consumed by a formula in an incompatible domain B.

        Only *computed* intermediates are checked — leaf/given variables
        (not produced by any formula in this trace) are always allowed.

        Returns True if coherent, False if a violation is found.
        """
        # variable -> domain of the formula that produces it in this trace
        produced_domain: Dict[str, str] = {
            he["output"]: he.get("domain", "unknown")
            for he in execution_order
        }

        for formula in execution_order:
            consumer_domain = formula.get("domain", "unknown")
            for inp in formula["inputs"]:
                if inp not in produced_domain:
                    continue  # leaf node — always OK
                producer_domain = produced_domain[inp]
                if producer_domain == consumer_domain:
                    continue
                allowed = self._DOMAIN_COMPAT.get(producer_domain, {"unknown"})
                if consumer_domain not in allowed:
                    logger.debug(
                        f"Domain coherence violation: '{formula['id']}' "
                        f"(domain={consumer_domain}) consumes '{inp}' "
                        f"produced by domain='{producer_domain}'. Rejecting trace."
                    )
                    return False
        return True

    def _get_execution_order(self, trace_path: List[Dict]) -> Optional[List[Dict]]:
        """
        Get the correct execution order for formulas using topological sort.
        
        Ensures that formulas execute in dependency order - a formula that produces
        a variable must execute before any formula that uses that variable.
        
        Args:
            trace_path: List of hyperedges (formulas) in the trace
            
        Returns:
            List of hyperedges in correct execution order, or None if there's a cycle
        """
        if not trace_path:
            return []
        
        # Build index mapping and dependency graph using indices (not dicts)
        index_to_hyperedge = {i: he for i, he in enumerate(trace_path)}
        
        # Build dependency graph: output -> index of formula that produces it
        output_to_index = {}
        for i, hyperedge in enumerate(trace_path):
            output_to_index[hyperedge['output']] = i
        
        # Build dependency graph: index -> set of indices it depends on
        index_dependencies = {}  # index -> set of indices it depends on
        
        for i, hyperedge in enumerate(trace_path):
            dependencies = set()
            for input_var in hyperedge['inputs']:
                if input_var in output_to_index:
                    # This input is produced by another formula in the trace
                    dependencies.add(output_to_index[input_var])
            index_dependencies[i] = dependencies
        
        # Topological sort using indices
        execution_order_indices = []
        remaining_indices = set(range(len(trace_path)))
        in_degree = {i: len(index_dependencies[i]) for i in range(len(trace_path))}
        
        # Find formulas with no dependencies (can execute first)
        queue = [i for i in range(len(trace_path)) if in_degree[i] == 0]
        
        while queue:
            # Execute a formula with no remaining dependencies
            current_idx = queue.pop(0)
            execution_order_indices.append(current_idx)
            remaining_indices.discard(current_idx)
            
            # Update dependencies: remove current from other formulas' dependencies
            for idx in remaining_indices:
                if current_idx in index_dependencies[idx]:
                    in_degree[idx] -= 1
                    if in_degree[idx] == 0:
                        queue.append(idx)
        
        # If we couldn't execute all formulas, there's a cycle
        if remaining_indices:
            remaining_formulas = [index_to_hyperedge[i]['id'] for i in remaining_indices]
            logger.debug(
                f"Circular dependency detected in trace. "
                f"Remaining formulas: {remaining_formulas}"
            )
            return None
        
        # Convert indices back to hyperedges
        execution_order = [index_to_hyperedge[i] for i in execution_order_indices]
        return execution_order
    
    def _complete_trace(self, trace: Dict, target: str) -> Optional[Dict]:
        """
        Complete a trace by adding missing required inputs to leaf_nodes.
        
        Validates that formulas can execute in dependency order. If a variable
        is used before it's calculated, the trace is rejected (returns None).
        
        Args:
            trace: Solution trace to complete
            target: Target variable being calculated
            
        Returns:
            Completed trace with missing inputs added to leaf_nodes, or None if
            trace has invalid execution order (variable used before calculation)
        """
        trace_path = trace['trace_path']
        leaf_nodes = trace['leaf_nodes'].copy()
        cycle_nodes = trace['cycle_nodes'].copy()
        
        if not trace_path:
            return trace
        
        # Get correct execution order using topological sort
        execution_order = self._get_execution_order(trace_path)
        if execution_order is None:
            # Circular dependency - reject trace
            logger.debug(f"Rejecting trace: circular dependency detected")
            return None

        # Reject cross-domain absurd traces (e.g. height from energy fed into geometry)
        if not self._check_domain_coherence(execution_order):
            return None

        # Track which variables are calculated by formulas in this trace
        calculated_outputs = {he['output'] for he in trace_path}
        
        # Validate execution order and add missing inputs
        available_vars_execution = leaf_nodes | cycle_nodes | {target}
        missing_inputs = set()
        
        for hyperedge in execution_order:
            # Check that all inputs for this formula are available
            for input_var in hyperedge['inputs']:
                if input_var not in available_vars_execution:
                    # Check if this input is produced by a formula that executes later
                    if input_var in calculated_outputs:
                        # This variable IS calculated in this trace, but after this formula
                        # This is invalid - variable used before calculation
                        logger.debug(
                            f"Rejecting trace: formula {hyperedge['id']} uses '{input_var}' "
                            f"which is calculated later in the execution order"
                        )
                        return None
                    else:
                        # This input is not produced by any formula - add to leaf_nodes
                        missing_inputs.add(input_var)
                        leaf_nodes.add(input_var)
                        available_vars_execution.add(input_var)
                        logger.debug(
                            f"Added missing input '{input_var}' to leaf_nodes for formula {hyperedge['id']}"
                        )
            
            # Add this formula's output to available variables for subsequent formulas
            available_vars_execution.add(hyperedge['output'])
        
        # Update trace_path to be in correct execution order
        completed_trace = trace.copy()
        completed_trace['trace_path'] = execution_order
        completed_trace['leaf_nodes'] = leaf_nodes
        
        if missing_inputs:
            logger.debug(
                f"Completed trace: added {len(missing_inputs)} missing inputs to leaf_nodes: "
                f"{sorted(missing_inputs)}"
            )
        
        return completed_trace
    
    def find_all_traces(
        self,
        target: str,
        max_depth: int = 10,
        max_traces: int = 100
    ) -> List[Dict]:
        """
        Find all solution traces for a target variable.
        
        Uses DFS to explore all paths, detecting cycles and treating
        cycle-creating variables as leaf nodes. Missing required inputs are
        automatically added to leaf_nodes to ensure traces are complete.
        
        Traces are validated to ensure formulas execute in correct dependency order.
        If a variable is used before it's calculated, the trace is rejected.
        
        Args:
            target: Target variable to solve for
            max_depth: Maximum depth for traversal
            max_traces: Maximum number of traces to return
            
        Returns:
            List of valid solution traces, each containing:
            - trace_path: list of hyperedges (formulas) used, in correct execution order
            - leaf_nodes: set of leaf nodes (base inputs, including any missing inputs that were added)
            - cycle_nodes: set of nodes that create cycles (treated as leaf nodes)
            - depth: depth of the trace
            
        Note: 
        - If a trace is missing required inputs, those inputs are automatically
          added to leaf_nodes so the trace becomes complete and usable.
        - Traces where variables are used before they're calculated are rejected.
        """
        if target not in self.all_nodes:
            logger.warning(f"Target variable '{target}' not found in hypergraph")
            return []
        
        traces = []
        visited_paths: Set[Tuple[str, ...]] = set()  # Track visited paths to avoid duplicates
        
        def dfs(node: str, path: List[Dict], visited: Set[str], cycle_nodes: Set[str], depth: int):
            """Recursive DFS to find all traces."""
            if len(traces) >= max_traces:
                return
            
            # Check depth limit
            if depth >= max_depth:
                # Create trace with current path
                all_nodes_in_path = visited | {node}
                leaf_nodes = all_nodes_in_path - set(he['output'] for he in path) - cycle_nodes - {target}
                
                trace = {
                    'trace_path': path.copy(),
                    'leaf_nodes': leaf_nodes,
                    'cycle_nodes': cycle_nodes.copy(),
                    'depth': depth
                }
                path_key = tuple(sorted(he['id'] for he in path))
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    traces.append(trace)
                return
            
            # Get hyperedges that produce this node
            available_hyperedges = self.output_to_hyperedges.get(node, [])
            
            if not available_hyperedges:
                # This is a leaf node (no formula to produce it)
                all_nodes_in_path = visited | {node}
                leaf_nodes = all_nodes_in_path - set(he['output'] for he in path) - cycle_nodes - {target}
                
                trace = {
                    'trace_path': path.copy(),
                    'leaf_nodes': leaf_nodes,
                    'cycle_nodes': cycle_nodes.copy(),
                    'depth': depth
                }
                path_key = tuple(sorted(he['id'] for he in path))
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    traces.append(trace)
                return
            
            # Try each hyperedge
            for hyperedge in available_hyperedges:
                # IMPORTANT: Skip hyperedges where target appears as input
                # The target is what we're calculating - it can't be a given input!
                if target in hyperedge['inputs']:
                    continue
                
                new_path = path + [hyperedge]
                new_visited = visited | {node}
                new_cycle_nodes = cycle_nodes.copy()
                
                # Check for cycles: if any input is already in visited_nodes, it creates a cycle
                # IMPORTANT: Never treat the target as a cycle node - it's what we're calculating!
                cycle_detected = False
                for input_var in hyperedge['inputs']:
                    if input_var in new_visited and input_var != node:
                        # This creates a cycle - treat input_var as a leaf node
                        # But never add the target itself to cycle_nodes (already checked above)
                        new_cycle_nodes.add(input_var)
                        cycle_detected = True
                
                # Collect all inputs that need to be available (non-cycle inputs)
                non_cycle_inputs = [inp for inp in hyperedge['inputs'] 
                                   if inp not in new_cycle_nodes]
                
                # Check if all non-cycle inputs are available (either in visited or will be explored)
                # Available means: already calculated (in visited), or will be explored further
                all_inputs_available = True
                for input_var in non_cycle_inputs:
                    if input_var not in new_visited:
                        # This input needs to be explored - mark as not fully available yet
                        # We'll create the trace later when we've explored all inputs
                        all_inputs_available = False
                        break
                
                # If cycle detected AND all non-cycle inputs are available, create trace
                if cycle_detected and all_inputs_available:
                    all_nodes_in_path = new_visited | set(hyperedge['inputs'])
                    leaf_nodes = all_nodes_in_path - set(he['output'] for he in new_path) - new_cycle_nodes - {target}
                    
                    trace = {
                        'trace_path': new_path.copy(),
                        'leaf_nodes': leaf_nodes,
                        'cycle_nodes': new_cycle_nodes.copy(),
                        'depth': depth + 1
                    }
                    path_key = tuple(sorted(he['id'] for he in new_path))
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        traces.append(trace)
                
                # Continue traversal for non-cycle inputs
                for input_var in hyperedge['inputs']:
                    if input_var not in new_visited:
                        # No cycle, continue exploring
                        dfs(input_var, new_path, new_visited, new_cycle_nodes, depth + 1)
                    elif input_var in new_visited and input_var != node:
                        # Cycle detected (already handled above if all inputs available)
                        pass
        
        # Start DFS from target
        dfs(target, [], set(), set(), 0)
        
        # Complete traces by adding missing inputs to leaf_nodes and validate execution order
        completed_traces = []
        for trace in traces:
            completed_trace = self._complete_trace(trace, target)
            if completed_trace is not None:
                # Trace is valid (no variables used before calculation)
                completed_traces.append(completed_trace)
            else:
                # Trace rejected due to invalid execution order
                logger.debug(
                    f"Rejected trace: invalid execution order. "
                    f"Formulas: {[he['id'] for he in trace['trace_path']]}"
                )
        
        # If no complete traces found, create a trace with target as leaf
        if not completed_traces:
            completed_traces.append({
                'trace_path': [],
                'leaf_nodes': {target},
                'cycle_nodes': set(),
                'depth': 0
            })
        
        return completed_traces
    
    def format_trace(self, trace: Dict) -> Dict:
        """
        Format a trace into a human-readable structure.
        
        Args:
            trace: Solution trace from find_all_traces
            
        Returns:
            Formatted trace with detailed information
        """
        formatted = {
            'depth': trace['depth'],
            'num_formulas': len(trace['trace_path']),
            'formulas': [],
            'leaf_nodes': sorted(list(trace['leaf_nodes'])),
            'cycle_nodes': sorted(list(trace['cycle_nodes'])),
            'calculation_steps': []
        }
        
        # Add formula details
        for i, hyperedge in enumerate(trace['trace_path']):
            step = {
                'step': i + 1,
                'output': hyperedge['output'],
                'formula': hyperedge['label'],
                'inputs': hyperedge['inputs'],
                'output_si_unit': hyperedge.get('output_si_unit', ''),
                'input_si_units': hyperedge.get('input_si_units', {})
            }
            formatted['formulas'].append(step)
            formatted['calculation_steps'].append(
                f"Step {i+1}: Calculate {hyperedge['output']} ({hyperedge.get('output_si_unit', '')}) "
                f"using {hyperedge['label']}"
            )
        
        return formatted
    
    def get_all_traces_formatted(self, target: str, max_depth: int = 10, max_traces: int = 100) -> List[Dict]:
        """
        Get all traces in formatted form.
        
        Args:
            target: Target variable
            max_depth: Maximum depth
            max_traces: Maximum number of traces
            
        Returns:
            List of formatted traces
        """
        traces = self.find_all_traces(target, max_depth, max_traces)
        return [self.format_trace(trace) for trace in traces]


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "acceleration"
    
    traverser = HypergraphTraverser("formula_hypergraph.json")
    traces = traverser.get_all_traces_formatted(target, max_depth=8, max_traces=20)
    
    print(f"\nFound {len(traces)} solution traces for target '{target}':\n")
    
    for i, trace in enumerate(traces, 1):
        print(f"{'='*80}")
        print(f"Trace {i}: Depth {trace['depth']}, {trace['num_formulas']} formulas")
        print(f"{'='*80}")
        
        if trace['leaf_nodes']:
            print(f"\nLeaf Nodes (given values): {', '.join(trace['leaf_nodes'])}")
        
        if trace['cycle_nodes']:
            print(f"Cycle Nodes (assumed given): {', '.join(trace['cycle_nodes'])}")
        
        print("\nCalculation Steps:")
        for step in trace['calculation_steps']:
            print(f"  {step}")
        
        print("\nDetailed Formula Information:")
        for formula_info in trace['formulas']:
            print(f"\n  Step {formula_info['step']}: {formula_info['output']} ({formula_info['output_si_unit']})")
            print(f"    Formula: {formula_info['formula']}")
            print(f"    Inputs: {', '.join(formula_info['inputs'])}")
            if formula_info['input_si_units']:
                input_units = [f"{inp} ({unit})" for inp, unit in formula_info['input_si_units'].items()]
                print(f"    Input Units: {', '.join(input_units)}")
        
        print("\n")