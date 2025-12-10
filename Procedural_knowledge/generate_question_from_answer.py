"""
Question Generation from Answer using Qwen LLM

This script takes the calculated answer from tree_walk_calculation.py and generates
a natural word problem question based on that answer using Qwen LLM.

Usage:
    # Option 1: Run standalone (runs calculation and generates question)
    python generate_question_from_answer.py
    
    # Option 2: Use as a module
    from generate_question_from_answer import QuestionGenerator
    from tree_walk_calculation import TreeWalkCalculator
    
    calculator = TreeWalkCalculator("variable_concept_graph.json", max_length=4)
    result = calculator.run("magnetic_flux", min_val=1.0, max_val=100.0)
    
    generator = QuestionGenerator()
    question = generator.generate_question(calculator)

Requirements:
    - transformers library: pip install transformers torch
    - Qwen model will be downloaded automatically on first use
"""

import logging
from typing import Optional, Dict, List
from tree_walk_calculation import TreeWalkCalculator

# Try to import transformers for Qwen LLM
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Question generation will be disabled.")
    print("Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('question_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate questions from calculated answers using Qwen LLM."""
    
    def __init__(self):
        """Initialize the QuestionGenerator."""
        self.llm_pipeline = None
        self._llm_initialized = False
    
    def _initialize_llm(self):
        """Lazy initialization of Qwen LLM."""
        if self._llm_initialized:
            return
        
        self._llm_initialized = True
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers library not available. Install with: pip install transformers torch")
            return
        
        try:
            logger.info("Loading Qwen LLM model for question generation (this may take a moment)...")
            self.llm_pipeline = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-3B-Instruct",
                device_map="auto",
                model_kwargs={"torch_dtype": "auto"}
            )
            logger.info("Qwen LLM model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Qwen LLM model: {e}")
            logger.warning("Question generation will be disabled.")
            self.llm_pipeline = None
    
    def _format_calculation_summary(self, calculator: TreeWalkCalculator) -> Optional[str]:
        """Format the calculation results into a summary string for LLM."""
        target = calculator.tree_structure['target']
        target_value = calculator.values.get(target, None)
        
        if target_value is None:
            return None
        
        summary_lines = []
        summary_lines.append(f"Calculation Summary:")
        summary_lines.append(f"Target Variable: {target}")
        summary_lines.append(f"Final Answer: {target_value:.4f}")
        summary_lines.append("")
        
        # Add formulas used
        if 'node_formulas' in calculator.tree_structure:
            summary_lines.append("Formulas Used:")
            for node, (formula, deps) in calculator.tree_structure['node_formulas'].items():
                if formula:
                    summary_lines.append(f"  {node} = {formula}")
            summary_lines.append("")
        
        # Add given values (leaf nodes)
        summary_lines.append("Given Values:")
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                summary_lines.append(f"  {leaf} = {calculator.values[leaf]:.4f}")
        summary_lines.append("")
        
        # Add calculated intermediate values
        summary_lines.append("Calculated Intermediate Values:")
        for level in sorted(calculator.tree_structure['levels'].keys()):
            for node in sorted(calculator.tree_structure['levels'][level]):
                if node not in calculator.tree_structure['leaf_nodes'] and node in calculator.values:
                    summary_lines.append(f"  {node} = {calculator.values[node]:.4f}")
        
        return "\n".join(summary_lines)
    
    def generate_question(self, calculator: TreeWalkCalculator) -> Optional[str]:
        """
        Generate a question based on the calculated answer using Qwen LLM.
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            
        Returns:
            Generated question string, or None if LLM is not available or generation fails.
        """
        # Initialize LLM if not already done
        self._initialize_llm()
        
        if not self.llm_pipeline:
            logger.warning("Qwen LLM not available. Cannot generate question.")
            return None
        
        target = calculator.tree_structure['target']
        target_value = calculator.values.get(target, None)
        
        if target_value is None:
            logger.warning("No target value calculated. Cannot generate question.")
            return None
        
        # Create prompt for question generation
        # DO NOT include the final answer - only given values and formulas
        system_prompt = """You are an expert physics/mathematics educator. Your task is to generate a natural, realistic word problem question.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. The question must be written in natural, conversational language
2. You MUST use ONLY the variables and values provided in the "Given Values" list below
3. You MUST include the EXACT numeric values for ALL given variables in your question
4. You MUST NOT use vague descriptions like "certain", "a distance", "some value" - use the actual numbers
5. You MUST include ALL given values with their exact numbers in the problem statement
6. You MUST NOT mention any variables that are NOT in the given values list
7. Ask for the target variable (what needs to be calculated)
8. Be appropriate for a physics or mathematics context
9. DO NOT include the final answer or any calculated values in the question
10. DO NOT mention the target variable's value or any intermediate calculated values
11. DO NOT invent or add any variables not in the given list
12. Be clear and well-structured
13. End with asking what the target variable is (e.g., "What is the impedance?")

EXAMPLE FORMAT:
If given values are: charge = 7.1454, length = 44.8581, time = 13.7825
Good question: "A circuit has a charge of 7.1454 Coulombs flowing through it. The circuit has a length of 44.8581 meters and operates for 13.7825 seconds. What is the impedance?"
Bad question: "A circuit has certain properties and some charge flowing through it. What is the impedance?" (missing exact values)

Generate ONLY the question text, without any explanation, answer, or numerical result. The question must include ALL given values with their exact numbers."""
        
        # Extract only given values (leaf nodes) for the prompt, NOT calculated values
        given_values_list = []
        given_values_dict = {}
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                value = calculator.values[leaf]
                given_values_list.append(leaf)
                given_values_dict[leaf] = value
        
        # Format given values clearly
        given_values_text = "Given Values (YOU MUST USE ONLY THESE VARIABLES AND VALUES):\n"
        for leaf in sorted(given_values_list):
            given_values_text += f"  {leaf} = {calculator.values[leaf]:.4f}\n"
        
        # Create a strict list of allowed variable names
        allowed_variables_text = f"\nALLOWED VARIABLE NAMES (use only these): {', '.join(sorted(given_values_list))}\n"
        allowed_variables_text += f"TARGET VARIABLE TO FIND: {target}\n"
        
        # Get formulas for context (but don't include calculated results)
        formulas_text = ""
        if 'node_formulas' in calculator.tree_structure:
            formulas_text = "\nFormulas that can be used (for context only, DO NOT include calculated results):\n"
            for node, (formula, deps) in calculator.tree_structure['node_formulas'].items():
                if formula:
                    formulas_text += f"  {node} = {formula}\n"
        
        # Format given values more explicitly for the LLM
        values_examples = []
        for var in sorted(given_values_list):
            value = given_values_dict[var]
            values_examples.append(f"{var} = {value:.4f}")
        values_list_text = "\n".join([f"  - {ex}" for ex in values_examples])
        
        user_prompt = f"""Generate a word problem question using ONLY the following information:

TARGET VARIABLE TO FIND: {target}

YOU MUST USE THESE EXACT VALUES (include ALL of them with their exact numbers):
{values_list_text}

ALLOWED VARIABLE NAMES (use ONLY these, no others): {', '.join(sorted(given_values_list))}

{formulas_text}

STRICT REQUIREMENTS - READ CAREFULLY:
1. You MUST include ALL {len(given_values_list)} given values in your question
2. You MUST write the EXACT numeric value for each variable (e.g., "charge of 7.1454 Coulombs", "length of 44.8581 meters")
3. You MUST NOT use vague language like "certain", "some", "a distance", "various" - use the actual numbers
4. You MUST NOT mention any variables NOT in the allowed list: {', '.join(sorted(given_values_list))}
5. You MUST NOT invent or add any new variables or values
6. Ask what the {target} is (DO NOT provide the answer)
7. Do NOT reveal any calculated values, intermediate results, or the final answer
8. Do NOT mention intermediate calculated variables (like current, resistance, voltage, etc.) unless they are in the allowed list above

EXAMPLE OF WHAT TO DO:
Given: charge = 7.1454, length = 44.8581, time = 13.7825
Question: "An electrical circuit has a charge of 7.1454 Coulombs flowing through it. The circuit has a length of 44.8581 meters and operates for 13.7825 seconds. What is the impedance?"

Generate a natural word problem question that includes ALL given values with their exact numbers."""
        
        try:
            # Format prompt for Qwen2.5-Instruct
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate question
            result = self.llm_pipeline(
                full_prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
            )
            
            generated_text = result[0]["generated_text"]
            
            # Extract the generated part (after the prompt)
            question = generated_text[len(full_prompt):].strip()
            
            # Clean up the question
            question = question.split("<|im_end|>")[0].strip()
            
            if question:
                # Validate that question only uses allowed variables and includes all values
                question = self._validate_question(question, given_values_dict, target)
                
                # Check if all values are included
                missing_values = self._check_missing_values(question, given_values_dict)
                if missing_values:
                    logger.warning(f"Question is missing these required values: {missing_values}")
                    logger.warning("The question may not include all required numeric values.")
                
                logger.info(f"\n{'='*60}")
                logger.info("Generated Question:")
                logger.info(f"{'='*60}\n{question}\n")
                return question
            else:
                logger.warning("No question generated from LLM.")
                return None
                
        except Exception as e:
            logger.error(f"Error generating question with Qwen LLM: {e}")
            return None
    
    def _validate_question(self, question: str, allowed_variables: Dict[str, float], target: str) -> str:
        """
        Validate and clean the generated question to ensure it only uses allowed variables.
        
        Args:
            question: Generated question text
            allowed_variables: Dictionary of allowed variable names and their values
            target: Target variable name
            
        Returns:
            Validated question (may be modified to remove invalid variables)
        """
        import re
        
        # Get all variable names from the question (simple pattern matching)
        # Look for patterns like "variable_name = value" or "variable_name as value"
        question_lower = question.lower()
        allowed_var_names = set(allowed_variables.keys())
        allowed_var_names.add(target.lower())
        
        # Check for numeric values that don't match allowed values
        # Extract all numbers from the question
        numbers_in_question = re.findall(r'\d+\.?\d*', question)
        
        # Check if any numbers don't match allowed values (with some tolerance)
        allowed_values = set()
        for var, val in allowed_variables.items():
            # Round to 4 decimal places for comparison
            allowed_values.add(round(val, 4))
        
        # Warn if question contains variables or values not in allowed list
        warnings = []
        
        # Check for common variable names that might be hallucinated
        common_physics_vars = ['magnetic_field', 'angle', 'width', 'height', 'mass', 'velocity', 
                              'acceleration', 'displacement', 'force', 'voltage', 'current',
                              'resistance', 'power', 'energy', 'field', 'flux']
        
        for var in common_physics_vars:
            if var not in allowed_var_names:
                # Check if this variable name appears in question (case insensitive)
                pattern = rf'\b{var}\b'
                if re.search(pattern, question_lower, re.IGNORECASE):
                    warnings.append(f"Question mentions '{var}' which is not in allowed variables")
        
        # Check for values that don't match allowed values
        for num_str in numbers_in_question:
            try:
                num_val = float(num_str)
                rounded_val = round(num_val, 4)
                # Check if this value is close to any allowed value (within 0.0001)
                found_match = False
                for allowed_val in allowed_values:
                    if abs(rounded_val - allowed_val) < 0.0001:
                        found_match = True
                        break
                if not found_match and rounded_val > 0.1:  # Ignore very small numbers
                    warnings.append(f"Question contains value '{num_val}' which doesn't match any allowed value")
            except ValueError:
                pass
        
        if warnings:
            logger.warning("Question validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
            logger.warning("The question may contain variables or values not in the tree walk.")
        
        return question
    
    def _check_missing_values(self, question: str, required_values: Dict[str, float]) -> List[str]:
        """
        Check if the question includes all required numeric values.
        
        Args:
            question: Generated question text
            required_values: Dictionary of required variable names and their values
            
        Returns:
            List of variable names whose values are missing from the question
        """
        import re
        missing = []
        
        # Extract all numbers from the question
        numbers_in_question = re.findall(r'\d+\.?\d*', question)
        numbers_set = {float(n) for n in numbers_in_question}
        
        # Check each required value
        for var, value in required_values.items():
            rounded_value = round(value, 4)
            # Check if this value (or close approximation) appears in the question
            found = False
            for num in numbers_set:
                if abs(round(num, 4) - rounded_value) < 0.0001:
                    found = True
                    break
            
            if not found:
                missing.append(f"{var} = {value:.4f}")
        
        return missing


def main():
    """Main function to run calculation and generate question."""
    # Configuration
    graph_file = "variable_concept_graph.json"
    target_node = "impedance"
    max_length = 2
    min_val = 1.0
    max_val = 100.0
    
    # Create calculator and run calculation
    logger.info("="*60)
    logger.info("Running Tree Walk Calculation")
    logger.info("="*60)
    calculator = TreeWalkCalculator(graph_file, max_length=max_length)
    result = calculator.run(target_node, min_val=min_val, max_val=max_val)
    
    # Print summary
    calculator.print_summary()
    
    if result is None:
        logger.error("Calculation failed. Cannot generate question.")
        return
    
    # Generate question from answer
    logger.info("\n" + "="*60)
    logger.info("Generating Question from Answer")
    logger.info("="*60)
    generator = QuestionGenerator()
    question = generator.generate_question(calculator)
    
    # Save question to file if generated
    if question:
        output_file = "generated_question.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Target Variable: {target_node}\n")
            f.write(f"Answer: {result:.4f}\n\n")
            f.write("Generated Question:\n")
            f.write(question)
            f.write("\n\n")
            f.write("="*60 + "\n")
            f.write("Calculation Details:\n")
            f.write("="*60 + "\n")
            f.write(generator._format_calculation_summary(calculator))
        logger.info(f"\nQuestion saved to {output_file}")
    
    return result, question


if __name__ == "__main__":
    main()

