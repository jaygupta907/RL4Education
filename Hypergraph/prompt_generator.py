"""
Prompt generation utilities for hypergraph traces.
"""
import random
from typing import Dict, Tuple
from hypergraph_traverser import HypergraphTraverser


def generate_value() -> float:
    """
    Generate a random value between 1 and 10 with up to 2 decimal places.
    """
    return round(random.uniform(1.0, 10.0), 2)


def create_prompt(traverser: HypergraphTraverser, trace: Dict, target: str, tokenizer) -> Tuple[str, Dict]:
    """
    Create a prompt for question generation that reflects the specific 
    hypergraph trace and calculation steps.
    """
    # Extract leaf nodes (given values)
    leaf_nodes = trace.get('leaf_nodes', [])
    
    # Extract formulas (calculation steps)
    formulas = trace.get('formulas', [])
    
    # Format the list of given values with SI units and actual values
    values_examples = []
    generated_values = {}  # Store generated values for logging
    for idx, var in enumerate(leaf_nodes, 1):
        # Get SI unit from hypergraph if available
        # Leaf nodes are INPUT variables, so we need to check input_si_units
        si_unit = ""
        for hyperedge in traverser.hypergraph['hyperedges']:
            input_si_units = hyperedge.get('input_si_units', {})
            if var in input_si_units:
                si_unit = input_si_units[var]
                break
        
        unit_str = si_unit if (si_unit and si_unit.strip()) else "(unit not specified)"
        # Generate actual value instead of placeholder (random value between 1 and 10)
        value = generate_value()
        generated_values[var] = {"value": value, "unit": unit_str}
        values_examples.append(f"{idx}. {var} = {value} {unit_str}")
    
    values_list_text = "\n".join(values_examples)
    allowed_vars_list = ", ".join(leaf_nodes)
    
    # Extract calculation steps for context
    calculation_steps = trace.get('calculation_steps', [])
    formulas = trace.get('formulas', [])
    
    # Format calculation steps
    steps_text = ""
    if formulas:
        steps_list = []
        for i, formula_info in enumerate(formulas, 1):
            step_var = formula_info.get('output', '')
            step_formula = formula_info.get('formula', '')
            step_inputs = formula_info.get('inputs', [])
            step_unit = formula_info.get('output_si_unit', '')
            unit_str = f" ({step_unit})" if step_unit else ""
            inputs_str = ", ".join(step_inputs)
            steps_list.append(f"Step {i}: Calculate {step_var}{unit_str} using {step_formula} with inputs: {inputs_str}")
        steps_text = "\n".join(steps_list)
    
    # For Llama 8B Instruct, use the proper chat template format
    system_prompt = """You are a physics problem generator. Generate clear, realistic physics word problems """
    
    # Build user prompt with calculation steps
    if steps_text:
        user_prompt = f""" Given the following values: {values_list_text} and target variable: {target}

Calculation steps:
{steps_text}

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between  target variable: {target} and the variables {allowed_vars_list}"""

        print(user_prompt)
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
        "leaf_nodes": leaf_nodes,
        "cycle_nodes": trace.get('cycle_nodes', []),
        "num_formulas": len(formulas),
        "depth": trace.get('depth', 0),
        "trace_path": [f['output'] for f in formulas],
        "user_prompt": user_prompt,  # Store user prompt for logging
        "generated_values": generated_values,  # Store generated values for reference
    }
    
    return prompt_text, metadata

