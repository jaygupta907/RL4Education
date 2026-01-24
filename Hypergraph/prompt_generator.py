"""
Prompt generation utilities for hypergraph traces.
"""
from typing import Dict, Tuple
from hypergraph_traverser import HypergraphTraverser


def create_prompt(traverser: HypergraphTraverser, trace: Dict, target: str, tokenizer) -> Tuple[str, Dict]:
    """
    Create a prompt for question generation that reflects the specific 
    hypergraph trace and calculation steps.
    """
    # Extract leaf nodes (given values)
    leaf_nodes = trace.get('leaf_nodes', [])
    
    # Extract formulas (calculation steps)
    formulas = trace.get('formulas', [])
    
    # Format the list of given values with SI units
    values_examples = []
    for idx, var in enumerate(leaf_nodes, 1):
        # Get SI unit from hypergraph if available
        si_unit = ""
        for hyperedge in traverser.hypergraph['hyperedges']:
            if hyperedge['output'] == var:
                si_unit = hyperedge.get('output_si_unit', '')
                break
        
        unit_str = si_unit if (si_unit and si_unit.strip()) else "(unit not specified)"
        values_examples.append(f"{idx}. {var} = [value] {unit_str}")
    
    values_list_text = "\n".join(values_examples)
    allowed_vars_list = ", ".join(leaf_nodes)
    
    # Extract calculation steps for context
    calculation_steps = trace.get('calculation_steps', [])
    
    # For Llama 8B Instruct, use the proper chat template format
    system_prompt = """You are a physics problem generator. Generate clear, realistic physics word problems in English using exact numerical values provided.

Your task:
1. Use ONLY variables from the allowed variables list above
2. Use EXACT values from the given values with full precision
3. Create a realistic physical scenario that naturally incorporates all given variables
4. Do NOT include phrases like "Here is the problem:" - start directly with the problem
5. Do NOT use placeholder symbols - use the actual numeric values provided
6. Generate ONLY the problem text, no preamble or explanations
7. WRITE in plain English, no LaTeX, markdown, or Unicode symbols"""
    
    user_prompt = f"""Use EXACT numeric values from the provided list. Do NOT round, modify, or approximate any numbers.

Given values ({len(leaf_nodes)} total):
{values_list_text}

Allowed variables: {allowed_vars_list}

Target variable: {target}

Calculation steps ({len(formulas)} steps):
{chr(10).join(calculation_steps[:5])}  # Show first 5 steps

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
        "leaf_nodes": leaf_nodes,
        "cycle_nodes": trace.get('cycle_nodes', []),
        "num_formulas": len(formulas),
        "depth": trace.get('depth', 0),
        "trace_path": [f['output'] for f in formulas],
    }
    
    return prompt_text, metadata

