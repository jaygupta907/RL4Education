import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any

# --- Step 1: Define Data Structures & Formulas ---

# We define the formulas as standalone functions for clarity
def calc_final_velocity(vars: Dict[str, Any]) -> float:
    """v = a * t (assumes from rest)"""
    return vars['a'] * vars['t']

def calc_kinetic_energy(vars: Dict[str, Any]) -> float:
    """ke = 0.5 * m * v**2"""
    return 0.5 * vars['m'] * vars['v']**2

@dataclass
class Question:
    """Holds our structured question data."""
    id: str
    concept: str
    text: str
    variables: Dict[str, Any]
    formula: Callable[[Dict[str, Any]], float]
    answer: float
    unit: str

# --- Step 2: Create the "Easy Question" Bank ---

# Question A: Find Final Velocity
q_101 = Question(
    id="q_101",
    concept="Final Velocity",
    text="A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?",
    variables={"a": 2, "t": 5},
    formula=calc_final_velocity,
    answer=10.0,
    unit="m/s"
)

# Question B: Find Kinetic Energy
q_102 = Question(
    id="q_102",
    concept="Kinetic Energy",
    text="A 3 kg object is moving at 10 m/s. What is its kinetic energy?",
    variables={"m": 3, "v": 10},
    formula=calc_kinetic_energy,
    answer=150.0,
    unit="Joules"
)

# Our simple database
question_bank = {
    "q_101": q_101,
    "q_102": q_102
}

# --- Step 3: Define the Chaining Generator ---

def generate_chained_question(q_a: Question, q_b: Question, new_id: str, text_template: str, intermediate_var: str):
    """
    Generates a new, harder question by chaining two easier ones.
    
    q_a: The first question in the chain (e.g., find velocity)
    q_b: The second question (e.g., find KE)
    new_id: ID for the generated question
    text_template: A string template for the new question text
    intermediate_var: The variable name that links the two (e.g., 'v')
    """
    
    print(f"--- Generating New Question (ID: {new_id}) ---")
    
    # 1. Get all variables from both questions
    # We start with q_a's variables
    new_vars = q_a.variables.copy()
    
    # Add variables from q_b, *except* for the one we are calculating
    for key, val in q_b.variables.items():
        if key != intermediate_var:
            new_vars[key] = val
            
    print(f"Combined variables: {new_vars}") 

    intermediate_value = q_a.formula(q_a.variables)
    print(f"Calculated intermediate '{intermediate_var}': {intermediate_value}")


    final_calc_vars = q_b.variables.copy()
    

    final_calc_vars[intermediate_var] = intermediate_value
    print(f"Final calculation inputs: {final_calc_vars}") 
    new_answer = q_b.formula(final_calc_vars)
    print(f"Calculated final answer: {new_answer}")

    new_text = text_template.format(**new_vars)

    return Question(
        id=new_id,
        concept=f"{q_b.concept} from {q_a.concept}",
        text=new_text,
        variables=new_vars,
        formula=None,
        answer=new_answer,
        unit=q_b.unit
    )

TEMPLATE = "A {m} kg object accelerates from rest at {a} m/s² for {t} seconds. What is its final kinetic energy?"

# Generate the question
hard_question = generate_chained_question(
    q_a=q_101,                  
    q_b=q_102,                  
    new_id="q_201",
    text_template=TEMPLATE,
    intermediate_var="v"     
)

print("\n--- RESULT ---")
print(f"Generated Text: {hard_question.text}")
print(f"Verifiable Answer: {hard_question.answer} {hard_question.unit}")