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
    examples = """
CORRECT Examples (follow these):

Example 1 (CORRECT):
Given values: force = 5.20 N, displacement = 3.50 m, mass = 7.20 kg
Target: kinetic_energy
Question: A sled is pulled along level snow by a constant horizontal force of 5.20 N through a displacement of 3.50 m. The sled and payload have a total mass of 7.20 kg. First determine the work done by the pull, then use that work (all converted to kinetic) to find the kinetic_energy of the sled. Determine the kinetic_energy.

Example 2 (CORRECT):
Given values: mass = 3.60 kg, specific_heat = 4.50 J/(kg·K), initial_temperature = 5.20 K, heat = 8.50 J
Target: final_temperature
Question: An aluminum block of mass 3.60 kg and specific heat 4.50 J/(kg·K) starts at an initial temperature of 5.20 K and absorbs 8.50 J of heat with no losses. First compute the temperature rise from the heat input, then determine the final_temperature of the block. Calculate the final_temperature.


---

INCORRECT Examples (avoid these mistakes):

Example 4 (WRONG - rounded values):
Given values: force = 5.20 N, displacement = 3.50 m, mass = 7.20 kg
Target: kinetic_energy
Question: A sled is pulled by a force of 5 N through a displacement of 3.5 m. The mass is 7 kg. Find the kinetic_energy.
❌ WRONG: Values were rounded (5.20 → 5, 3.50 → 3.5, 7.20 → 7). Use EXACT values.

Example 5 (WRONG - missing variables):
Given values: mass = 3.60 kg, specific_heat = 4.50 J/(kg·K), initial_temperature = 5.20 K, heat = 8.50 J
Target: final_temperature
Question: An aluminum block of mass 3.60 kg starts at 5.20 K. Calculate the final_temperature.
❌ WRONG: Missing variables (specific_heat and heat). Use ALL provided variables.

Example 6 (WRONG - added extra values):
Given values: force = 5.20 N, displacement = 3.50 m
Target: work
Question: A force of 5.20 N acts through 3.50 m. The object has mass 2.00 kg. Find the work.
❌ WRONG: Added mass (2.00 kg) which was not in given values. Use ONLY variables from allowed_variables.
        """

    # For Llama 8B Instruct, use the proper chat template format
    system_prompt = """You are a physics problem generator. Generate clear, realistic physics word problems in English using exact numerical values provided.

Your task:
- Use ALL provided numeric values EXACTLY as given (no rounding, no modifications) otherwise you will be penalized heavily.
- Include proper SI units for each value
- Create a realistic physical scenario that naturally incorporates all given variables
- End by asking for the target variable
- Generate ONLY the problem text, no preamble or explanations"""
    
    user_prompt = f"""

<critical_instructions>

Use EXACT numeric values from the provided list.

Do NOT round, modify, or approximate any numbers.

</critical_instructions>

<given_values count="{len(given_values_list)}">

{values_list_text}

</given_values>

<allowed_variables>

{allowed_vars_list}

</allowed_variables>

<requirements priority="high">

1. Use ONLY variables from <allowed_variables>

2. Use EXACT values from <given_values> with full precision

3. End with asking for target variable"

4. Create a realistic physical scenario that naturally incorporates all given variables.

5. Do NOT include phrases like "Here is the problem:" - start directly with the problem.

6. Do NOT use placeholder symbols  - use the actual numeric values provided.

5. Generate ONLY the problem text, no preamble or explanations.

7. WRITE in plain English, no LaTeX, markdown, or Unicode symbols.

</requirements>

<examples>

{examples}

</examples>

Generate the problem now, strictly following <critical_instructions> and <requirements>.

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

