"""
Question Judge - Evaluates if generated questions correctly ask for the solution trace

This script uses a small judge LLM to evaluate whether a generated question correctly
asks for the solution trace based on the pruned tree walk calculation.

Usage:
    from question_judge import QuestionJudge
    from tree_walk_calculation import TreeWalkCalculator
    from generate_question_from_answer import QuestionGenerator
    
    calculator = TreeWalkCalculator("variable_concept_graph.json", max_length=4)
    result = calculator.run("magnetic_flux", min_val=1.0, max_val=100.0)
    
    generator = QuestionGenerator()
    question = generator.generate_question(calculator)
    
    judge = QuestionJudge()
    score = judge.evaluate(calculator, question)
    print(f"Question quality score: {score}/10")

Requirements:
    - transformers library: pip install transformers torch
    - Qwen 2.5 3B model will be downloaded automatically on first use
"""

import logging
from typing import Optional, Dict, Tuple
from tree_walk_calculation import TreeWalkCalculator

# Try to import transformers for Qwen LLM
try:
    from transformers import pipeline, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Question judging will be disabled.")
    print("Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('question_judge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QuestionJudge:
    """Judge LLM to evaluate if generated questions correctly ask for the solution trace."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Initialize the QuestionJudge.
        
        Args:
            model_name: Name of the judge LLM model to use
        """
        self.model_name = model_name
        self.judge_pipeline = None
        self.tokenizer = None
        self._judge_initialized = False
    
    def _initialize_judge(self):
        """Lazy initialization of judge LLM."""
        if self._judge_initialized:
            return
        
        self._judge_initialized = True
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers library not available. Install with: pip install transformers torch")
            return
        
        try:
            logger.info(f"Loading judge LLM model ({self.model_name})...")
            # Load tokenizer separately to use chat template
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.judge_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device_map="auto",
                model_kwargs={"torch_dtype": "auto"}
            )
            logger.info("Judge LLM model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load judge LLM model: {e}")
            logger.warning("Question judging will be disabled.")
            self.judge_pipeline = None
            self.tokenizer = None
    
    def _format_solution_trace(self, calculator: TreeWalkCalculator) -> str:
        """
        Format the solution trace from the pruned tree walk.
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            
        Returns:
            Formatted string describing the solution trace
        """
        target = calculator.tree_structure['target']
        target_value = calculator.values.get(target, None)
        
        if target_value is None:
            return "No solution trace available."
        
        trace_lines = []
        trace_lines.append(f"Solution Trace for: {target}")
        trace_lines.append("=" * 60)
        trace_lines.append(f"\nTarget Variable: {target}")
        trace_lines.append(f"Final Answer: {target_value:.4f}")
        trace_lines.append("\nGiven Values (Leaf Nodes):")
        
        # List all leaf nodes with their values
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                trace_lines.append(f"  • {leaf} = {calculator.values[leaf]:.4f}")
        
        trace_lines.append("\nCalculation Steps:")
        
        # Show calculation steps by level
        all_levels = sorted(calculator.tree_structure['levels'].keys())
        for level in all_levels:
            if level == 0:  # Skip target level
                continue
            level_nodes = calculator.tree_structure['levels'].get(level, [])
            for node in sorted(level_nodes):
                if node not in calculator.tree_structure['leaf_nodes'] and node in calculator.values:
                    # Show formula if available
                    formula_info = ""
                    if 'node_formulas' in calculator.tree_structure:
                        if node in calculator.tree_structure['node_formulas']:
                            formula, deps = calculator.tree_structure['node_formulas'][node]
                            if formula:
                                formula_info = f" (using: {formula})"
                    trace_lines.append(f"  Level {level}: {node} = {calculator.values[node]:.4f}{formula_info}")
        
        return "\n".join(trace_lines)
    
    def _check_required_values_present(self, calculator: TreeWalkCalculator, question: str) -> Dict:
        """
        Programmatically check if all required values from solution trace are present in question.
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            question: Generated question string
            
        Returns:
            Dictionary with check results: {'all_present': bool, 'missing': list, 'present': list, 'incorrect_values': list}
        """
        import re
        
        # Get required leaf node values
        leaf_values = {}
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                leaf_values[leaf] = calculator.values[leaf]
        
        # Extract all numbers from question
        numbers_in_question = re.findall(r'\d+\.?\d*', question)
        numbers_set = {float(n) for n in numbers_in_question}
        
        missing = []
        present = []
        incorrect_values = []
        
        # Check each required value
        for var, required_value in leaf_values.items():
            rounded_required = round(required_value, 4)
            found = False
            
            # Check if this value (or close approximation) appears in question
            for num in numbers_set:
                rounded_num = round(num, 4)
                if abs(rounded_num - rounded_required) < 0.0001:
                    found = True
                    present.append(var)
                    break
            
            if not found:
                missing.append(f"{var} = {required_value:.4f}")
        
        # Check for values that don't match any required value (potential incorrect values)
        required_values_set = {round(val, 4) for val in leaf_values.values()}
        for num in numbers_set:
            rounded_num = round(num, 4)
            if rounded_num > 0.1:  # Ignore very small numbers
                found_match = False
                for req_val in required_values_set:
                    if abs(rounded_num - req_val) < 0.0001:
                        found_match = True
                        break
                if not found_match:
                    incorrect_values.append(rounded_num)
        
        all_present = len(missing) == 0
        
        return {
            'all_present': all_present,
            'missing': missing,
            'present': present,
            'incorrect_values': incorrect_values,
            'total_required': len(leaf_values),
            'found_count': len(present)
        }
    
    def evaluate(self, calculator: TreeWalkCalculator, question: str) -> Optional[Dict]:
        """
        Evaluate if the generated question correctly asks for the solution trace.
        Uses LLM judge score only (no programmatic scoring).
        Returns score and explanation.
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            question: Generated question string
            
        Returns:
            Dictionary with 'score' (0.0-10.0) and 'explanation' (str), or None if judge LLM is not available
        """
        self._initialize_judge()
        
        if not self.judge_pipeline:
            logger.warning("Judge LLM not available. Cannot evaluate question.")
            return None
        
        if not question:
            logger.warning("No question provided for evaluation.")
            return {"score": 0.0, "explanation": "No question provided"}
        
        target = calculator.tree_structure['target']
        target_value = calculator.values.get(target, None)
        
        if target_value is None:
            logger.warning("No target value calculated. Cannot evaluate question.")
            return {"score": 0.0, "explanation": "No target value available"}
        
        # Format solution trace
        solution_trace = self._format_solution_trace(calculator)
        
        # Get leaf nodes and their values
        leaf_nodes = sorted(calculator.tree_structure['leaf_nodes'])
        leaf_values = {}
        for leaf in leaf_nodes:
            if leaf in calculator.values:
                leaf_values[leaf] = calculator.values[leaf]
        
        # Do programmatic check first
        check_result = self._check_required_values_present(calculator, question)
        
        # Create evaluation prompt focused on solution trace correctness with detailed feedback
        system_prompt = """You are an expert evaluator for physics/mathematics word problems. 
Your task is to evaluate if a generated question provides ALL the correct variables needed to calculate the target variable as shown in the solution trace.

CRITICAL: You must check EACH required value individually. Look for the EXACT number in the question text.
Do NOT assume values are present - verify by finding the actual number in the question.

CRITICAL EVALUATION CRITERIA:
1. Does the question include ALL required leaf node values from the solution trace? (Check each one)
2. Are the values EXACTLY correct (match the solution trace values)? (Verify numbers match)
3. Does the question ask for the correct target variable?
4. Would solving the question with the provided values lead to the EXACT same solution trace?

SCORING (0.0 to 10.0):
- 10.0: ALL required variables with EXACT correct values, correct target, would produce same solution trace
- 8.0-9.9: All required variables included, values mostly correct (minor rounding differences OK)
- 6.0-7.9: Most variables included, some values correct
- 4.0-5.9: Some variables missing or incorrect values
- 0.0-3.9: Missing critical variables or wrong values, cannot produce solution trace

Respond in this format:
SCORE: X.X
EXPLANATION: [detailed explanation listing which variables are present/missing and if values are correct]"""
        
        # Format required values list clearly
        required_values_text = "\n".join([f"{idx}. {var} = {val:.4f}" 
                                         for idx, (var, val) in enumerate(sorted(leaf_values.items()), 1)])
        
        # Include programmatic check results in prompt for context
        programmatic_info = ""
        if check_result['missing']:
            programmatic_info = f"\n⚠️ PROGRAMMATIC CHECK FOUND MISSING VALUES: {', '.join(check_result['missing'])}\n"
            programmatic_info += f"Found {check_result['found_count']} out of {check_result['total_required']} required values.\n"
        
        user_prompt = f"""Evaluate if this question provides ALL correct variables to calculate the target:

QUESTION:
{question}

SOLUTION TRACE (what the question should lead to):
{solution_trace}

REQUIRED VALUES (ALL {len(leaf_values)} must be in question with EXACT numbers):
{required_values_text}
{programmatic_info}
TARGET VARIABLE: {target}
EXPECTED ANSWER: {target_value:.4f}

INSTRUCTIONS:
1. Check EACH required value above - is its number present in the question?
2. Verify the numbers match exactly (within 0.0001 tolerance)
3. List which values are present and which are missing
4. Score based on completeness: missing values = lower score

Provide score and explanation:"""
        
        try:
            # Format prompt using Qwen 2.5 chat template
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                full_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback format if chat template not available (Qwen uses ChatML format)
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate evaluation with more tokens for detailed feedback
            result = self.judge_pipeline(
                full_prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
            )
            
            generated_text = result[0]["generated_text"]
            
            # Extract the generated part (after the prompt)
            response = generated_text[len(full_prompt):].strip()
            
            # Clean up the response (remove any remaining special tokens)
            response = response.split("<|im_end|>")[0].strip()
            response = response.split("<|end_of_text|>")[0].strip()
            
            # Parse score and explanation
            import re
            score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', response, re.IGNORECASE)
            explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=SCORE:|$)', response, re.IGNORECASE | re.DOTALL)
            
            score = None
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))
            
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            
            if score is not None:
                logger.info(f"\n{'='*60}")
                logger.info("Question Evaluation:")
                logger.info(f"{'='*60}")
                logger.info(f"LLM Judge Score: {score}/10.0")
                logger.info(f"{'='*60}\n")
                
                return {"score": score, "explanation": explanation}
            else:
                logger.warning(f"Could not extract score from judge response: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating question with judge LLM: {e}")
            return None


def main():
    """Main function to test the question judge."""
    from generate_question_from_answer import QuestionGenerator
    
    # Configuration
    graph_file = "variable_concept_graph.json"
    target_node = "stress"
    max_length = 2
    min_val = 1.0
    max_val = 100.0
    
    # Create calculator and run calculation
    logger.info("="*60)
    logger.info("Running Tree Walk Calculation")
    logger.info("="*60)
    calculator = TreeWalkCalculator(graph_file, max_length=max_length)
    result = calculator.run(target_node, min_val=min_val, max_val=max_val)
    
    if result is None:
        logger.error("Calculation failed. Cannot evaluate question.")
        return
    
    # Generate question
    logger.info("\n" + "="*60)
    logger.info("Generating Question")
    logger.info("="*60)
    generator = QuestionGenerator()
    question = generator.generate_question(calculator)
    
    if not question:
        logger.error("Question generation failed. Cannot evaluate.")
        return
    
    # Evaluate question
    logger.info("\n" + "="*60)
    logger.info("Evaluating Question")
    logger.info("="*60)
    judge = QuestionJudge()
    
    # Evaluate question
    result = judge.evaluate(calculator, question)
    if result:
        logger.info(f"\nFinal Score: {result['score']}/10.0")
        logger.info(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()

