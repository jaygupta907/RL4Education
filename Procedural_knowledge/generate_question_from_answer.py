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
                model="Qwen/Qwen2.5-7B-Instruct",
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
    
    def _get_questions_for_nodes(self, calculator: TreeWalkCalculator) -> Dict[str, List[str]]:
        """
        Extract questions for all nodes in the pruned tree walk (excluding target node).
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            
        Returns:
            Dictionary mapping variable names to their questions
        """
        questions_dict = {}
        
        # Get target node to exclude it from context
        target = calculator.tree_structure.get('target')
        
        # Get all nodes in the pruned tree walk
        all_nodes = calculator.tree_structure.get('nodes', set())
        
        # Extract questions for each node from the graph data (excluding target)
        for node in all_nodes:
            if node != target and node in calculator.variable_info:
                var_info = calculator.variable_info[node]
                if 'questions' in var_info and var_info['questions']:
                    questions_dict[node] = var_info['questions']
        
        return questions_dict
    
    def _format_questions_context(self, questions_dict: Dict[str, List[str]], calculator: TreeWalkCalculator) -> str:
        """
        Format questions from all nodes in the pruned tree walk (excluding target) as context for the LLM prompt.
        
        Args:
            questions_dict: Dictionary mapping variable names to their questions (target excluded)
            calculator: TreeWalkCalculator instance
            
        Returns:
            Formatted string with example questions
        """
        if not questions_dict:
            return ""
        
        context_lines = []
        context_lines.append("EXAMPLE QUESTIONS from all nodes in pruned tree walk (phrasing style only):")
        
        # Organize by level to show the calculation path
        all_levels = sorted(calculator.tree_structure.get('levels', {}).keys(), reverse=True)
        
        for level in all_levels:
            level_nodes = calculator.tree_structure['levels'].get(level, [])
            for node in sorted(level_nodes):
                if node in questions_dict:
                    node_questions = questions_dict[node]
                    context_lines.append(f"\n{node} (Level {level}):")
                    for i, q in enumerate(node_questions[:2], 1):  # Show max 2 questions per node
                        context_lines.append(f"  {i}. {q}")
        
        return "\n".join(context_lines)
    
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
        
        # Get questions for all nodes in the pruned tree walk
        questions_dict = self._get_questions_for_nodes(calculator)
        
        # Identify leaf vs intermediate nodes
        leaf_nodes_set = calculator.tree_structure.get('leaf_nodes', set())
        all_nodes_set = calculator.tree_structure.get('nodes', set())
        intermediate_nodes = sorted([n for n in all_nodes_set if n not in leaf_nodes_set and n != target])
        
        # Log which nodes have questions
        if questions_dict:
            logger.info(f"\n{'='*60}")
            logger.info("Question generation context:")
            logger.info(f"{'='*60}")
            logger.info(f"LEAF NODES (will be used IN the question): {sorted(leaf_nodes_set)}")
            logger.info(f"INTERMEDIATE NODES (context only, NOT in question): {intermediate_nodes}")
            logger.info(f"\nFound example questions for nodes in pruned tree walk:")
            leaf_with_questions = [n for n in sorted(questions_dict.keys()) if n in leaf_nodes_set]
            intermediate_with_questions = [n for n in sorted(questions_dict.keys()) if n not in leaf_nodes_set]
            if leaf_with_questions:
                logger.info(f"  Leaf nodes with examples: {leaf_with_questions}")
            if intermediate_with_questions:
                logger.info(f"  Intermediate nodes with examples: {intermediate_with_questions}")
            logger.info(f"{'='*60}\n")
        else:
            logger.info("No example questions found in the pruned tree walk nodes.")
        
        questions_context = self._format_questions_context(questions_dict, calculator)
        
        # Print the formatted questions context for debugging/inspection
        if questions_context:
            logger.info(f"\n{'='*60}")
            logger.info("Questions Context (formatted for LLM):")
            logger.info(f"{'='*60}")
            logger.info(questions_context)
            logger.info(f"{'='*60}\n")
        
        # System prompt - concise and direct
        system_prompt = """Generate physics/mathematics word problems in English only.

Rules:
• Use EXACT numeric values from the trace - no rounding or approximations
• Write in plain English (ASCII only) - no LaTeX, markdown, or Unicode symbols
• Include all given values with their exact numbers
• End with "What is the [target]?"
• English only - no other languages"""
        
        # Extract only given values (leaf nodes) for the prompt, NOT calculated values
        given_values_list = []
        given_values_dict = {}
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                value = calculator.values[leaf]
                given_values_list.append(leaf)
                given_values_dict[leaf] = value
        
        # Format given values clearly with full precision
        given_values_text = "Given Values (YOU MUST USE ONLY THESE VARIABLES AND EXACT VALUES):\n"
        for leaf in sorted(given_values_list):
            given_values_text += f"  {leaf} = {calculator.values[leaf]:.10f}\n"
        
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
        
        # Format given values more explicitly for the LLM with clear numbering and full precision
        values_examples = []
        for idx, var in enumerate(sorted(given_values_list), 1):
            value = given_values_dict[var]
            values_examples.append(f"{idx}. {var} = {value:.10f}")
        values_list_text = "\n".join(values_examples)
        
        # Create a clear list of allowed variables
        allowed_vars_list = ", ".join(sorted(given_values_list))
        
        # Identify leaf vs intermediate nodes for clarity in prompt
        leaf_nodes_set = calculator.tree_structure.get('leaf_nodes', set())
        all_nodes_set = calculator.tree_structure.get('nodes', set())
        intermediate_nodes = sorted([n for n in all_nodes_set if n not in leaf_nodes_set and n != target])
        
        # User prompt
        user_prompt = f"""Generate a physics/mathematics word problem question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Find the {target}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU MUST USE THESE {len(given_values_list)} VALUES (include ALL with exact numbers):

{values_list_text}

ALLOWED VARIABLES ONLY: {allowed_vars_list}

Do NOT use: {', '.join(intermediate_nodes[:5]) if intermediate_nodes else 'any calculated/intermediate variables'}

{formulas_text if formulas_text else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE QUESTIONS (for phrasing style only):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{questions_context if questions_context else "No examples available."}

⚠️ Note: Examples show phrasing style. Use ONLY variables from the allowed list above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIREMENTS:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Include ALL {len(given_values_list)} values above with their EXACT numbers

2. Use ONLY these variables: {allowed_vars_list}

3. Write in natural language with appropriate units

4. End with: "What is the {target}?"

5. Do NOT use vague terms like "certain", "some", "a distance" - use actual numbers

6. Do NOT mention intermediate/calculated variables

Generate the question now:"""
        
        try:
            # Format prompt for Qwen2.5-Instruct
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate question
            result = self.llm_pipeline(
                full_prompt,
                max_new_tokens=400,
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
            )
            
            generated_text = result[0]["generated_text"]
            
            # Extract the generated part (after the prompt)
            question = generated_text[len(full_prompt):].strip()
            
            # Clean up the question
            question = question.split("<|im_end|>")[0].strip()
            
            if question:
                
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
    target_node = "magnetic_flux"
    max_length = 3
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

