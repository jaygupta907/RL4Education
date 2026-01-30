"""
Prompt generation utilities.
"""
from typing import Dict, Tuple
from tree_walk_calculation import TreeWalkCalculator


def create_prompt(calculator: TreeWalkCalculator, tokenizer) -> Tuple[str, Dict]:
    """
    Create a prompt for question generation that reflects the specific 
    tree walk trace and intermediate steps.
    """
    target = calculator.tree_structure['target']
    leaf_nodes_set = calculator.tree_structure.get('leaf_nodes', set())
    all_nodes_set = calculator.tree_structure.get('nodes', set())
    
    # Identify given values (leaf nodes)
    given_values_list = sorted([leaf for leaf in leaf_nodes_set if leaf in calculator.values])
    
    # Identify intermediate steps (nodes that are neither leaf nor target)
    # These represent the "path" the calculator took through the random formulas
    intermediate_nodes = sorted([n for n in all_nodes_set if n not in leaf_nodes_set and n != target])

    # Format the list of given values with exact numbers and SI units
    values_examples = []
    for idx, var in enumerate(given_values_list, 1):
        value = calculator.values[var]
        si_unit = calculator._get_si_unit(var)
        unit_str = si_unit if (si_unit and si_unit.strip()) else "(unit not specified)"
        values_examples.append(f"{idx}. {var} = {value} {unit_str}")
    
    values_list_text = "\n".join(values_examples)
    allowed_vars_list = ", ".join(given_values_list)
    
    # Extract calculation steps from tree structure
    steps_text = ""
    if 'levels' in calculator.tree_structure and 'node_formulas' in calculator.tree_structure:
        steps_list = []
        all_levels = sorted([l for l in calculator.tree_structure['levels'].keys() if l > 0])  # Skip level 0 (target)
        
        step_num = 1
        for level in all_levels:
            level_nodes = calculator.tree_structure['levels'].get(level, [])
            for node in sorted(level_nodes):
                if node not in leaf_nodes_set and node in calculator.tree_structure['node_formulas']:
                    formula, deps = calculator.tree_structure['node_formulas'][node]
                    if formula:
                        si_unit = calculator._get_si_unit(node)
                        unit_str = f" ({si_unit})" if si_unit else ""
                        inputs_str = ", ".join(sorted(deps)) if deps else ""
                        steps_list.append(f"Step {step_num}: Calculate {node}{unit_str} using {formula} with inputs: {inputs_str}")
                        step_num += 1
        
        if steps_list:
            steps_text = "\n".join(steps_list)

    # For Llama 8B Instruct, use the proper chat template format
    system_prompt = """You are a physics problem generator. Generate clear, realistic physics word problems """
    
    # Build user prompt with calculation steps
    if steps_text:
        user_prompt = f""" Given the following values: {values_list_text} and target variable: {target}

Calculation steps:
{steps_text}

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between  target variable: {target} and the variables {allowed_vars_list}"""

    else:
        user_prompt = f""" Given the following values: {values_list_text} and allowed variables: {allowed_vars_list} Generate a deep-reasoning physics question that tests a student's understanding of the relationship between {target} and the variables {allowed_vars_list}"""
    
    # Apply Llama 3 Chat Template
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback manual format for Llama 3
            prompt_text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
    else:
        prompt_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    metadata = {
        "target": target,
        "leaf_nodes": given_values_list,
        "intermediate_nodes": intermediate_nodes,
        "trace_path": [n for n in calculator.tree_structure.get('nodes', [])],
    }
    
    return prompt_text, metadata

