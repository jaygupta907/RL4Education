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

    # For Llama 8B Instruct, use the proper chat template format
    system_prompt = """You are a physics problem generator. Generate clear, realistic physics word problems in English using exact numerical values provided.

Your task:
1. Use ONLY variables from the allowed variables list above
2. Use EXACT values from the given values with full precision
4. Create a realistic physical scenario that naturally incorporates all given variables
5. Do NOT include phrases like "Here is the problem:" - start directly with the problem
6. Do NOT use placeholder symbols - use the actual numeric values provided
7. Generate ONLY the problem text, no preamble or explanations
8. WRITE in plain English, no LaTeX, markdown, or Unicode symbols"""
    
    user_prompt = f"""Use EXACT numeric values from the provided list. Do NOT round, modify, or approximate any numbers.

Given values ({len(given_values_list)} total):
{values_list_text}

Allowed variables: {allowed_vars_list}

Target variable: {target}

Generate the problem now, strictly following the requirements above.
"""
    
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

