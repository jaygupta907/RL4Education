import random
from sympy import symbols, Eq, solve
from transformers import pipeline  # Added for local LLM

# --- 1. Define Symbolic Variables ---
# We define all the symbols we plan to use in our physics equations.
F, m, a, v_f, v_i, t = symbols("F m a v_f v_i t")



force_template = {
    "name": "force",
    "equation": Eq(F, m * a),
    "variables": {"F": "Force (N)", "m": "mass (kg)", "a": "acceleration (m/s²)"},
    "value_pools": {
        "m": [1000, 1200, 1500, 2000],
        "a": [2, 2.5, 3, 4, 5],
        "F": [3000, 4000, 5000, 6000],
    },
    "text_templates": {
        "F": "A car of mass {m} kg accelerates at {a} m/s². What is the net force (N) acting on it?",
        "m": "A net force of {F} N is applied to a car, causing it to accelerate at {a} m/s². What is its mass (kg)?",
        "a": "A net force of {F} N is applied to a {m} kg car. What is its acceleration (m/s²)?",
    },
}

kinematics_template = {
    "name": "kinematics_vf_vi_at",
    "equation": Eq(v_f, v_i + a * t),
    "variables": {
        "v_f": "final velocity (m/s)",
        "v_i": "initial velocity (m/s)",
        "a": "acceleration (m/s²)",
        "t": "time (s)",
    },
    "value_pools": {
        "v_i": [0, 10, 20],
        "a": [2, 3, 4, 5],  
        "t": [5, 10, 12, 15],
    },
    "text_templates": {
        "v_f": "A {object_name} starting at {v_i} m/s accelerates at {a} m/s² for {t} seconds. What is its final velocity (m/s)?"
    },
}

TEMPLATE_DATABASE = {"force": force_template, "kinematics": kinematics_template}


MISC_VALUE_POOLS = {"object_name": ["train", "car", "rocket", "object"]}


class ProblemSolver:
    """
    Analyzes a template database to build a "solver map" and can then
    find a path from a set of "given" variables to a "goal" variable.
    """

    def __init__(self, template_db):
        self.template_db = template_db
        self.solver_map = self._build_solver_map()
        self._build_value_pools_and_maps()  

        print("Loading LLM model (this may take a moment)...")
        try:
            self.llm_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
            print("LLM model loaded.")
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            print(
                "Please ensure 'transformers' and 'torch' are installed: pip install transformers torch"
            )
            self.llm_pipeline = None

    def _build_solver_map(self):
        """
        Analyzes the template DB and builds a map of:
        {goal_variable: [list of ways to solve for it]}

        Example:
        {a: [
              {'template': 'force', 'formula': F/m, 'inputs': {F, m}},
              {'template': 'kinematics', 'formula': (v_f-v_i)/t, 'inputs': {v_f, v_i, t}}
            ],
         ...
        }
        """
        solver_map = {}
        for name, template in self.template_db.items():
            equation = template["equation"]
            variables_in_eq = equation.free_symbols

            for var in variables_in_eq:
                if var not in solver_map:
                    solver_map[var] = []

                try:

                    formula = solve(equation, var)[0]
                    formula_eq = Eq(var, formula)
                    inputs = formula.free_symbols

                    solver_map[var].append(
                        {
                            "template_name": name,
                            "template": template,
                            "formula": formula_eq,  
                            "inputs": inputs,
                        }
                    )
                except IndexError:
                    pass
        return solver_map

    def _build_value_pools_and_maps(self):
        """Combines all value pools and creates a variable description map."""
        self.all_pools = MISC_VALUE_POOLS.copy()
        self.var_details = {}  # NEW: {m: "mass (kg)", F: "Force (N)", ...}
        for template in self.template_db.values():
            self.all_pools.update(template["value_pools"])
            # Build the var_details map from the 'variables' section
            for var_str, var_desc in template["variables"].items():
                var_sym = symbols(var_str)
                if var_sym not in self.var_details:
                    self.var_details[var_sym] = var_desc

    def _find_solution_path(self, givens, goal, plan, known_vars):
        """
        Recursive "pathfinding" function (depth-first search)
        to find a way from 'givens' to 'goal'.
        """
        # 1. Base Case: We already have this variable
        if goal in known_vars:
            return True

        # 2. Find a way to solve for the goal
        if goal not in self.solver_map:
            return False  # This variable is unsolvable

        # 3. Try every possible formula for solving for 'goal'
        for solver_option in self.solver_map[goal]:
            inputs = solver_option["inputs"]

            # Check if we can find all inputs for this formula
            if all(
                self._find_solution_path(givens, inp, plan, known_vars)
                for inp in inputs
            ):
                # We found a valid path!
                # Add this step to our plan
                plan.append(solver_option)
                # This goal is now "known"
                known_vars.add(goal)
                return True

        # 4. We tried all options and none worked
        return False

    def generate_question(self, givens, goal):  # No longer async
        """
        Generates a multi-step question by dynamically finding a
        solution path and using an LLM to write the problem text.
        """
        print(f"--- Generating (Method 3: Dynamic Composition + LLM Text) ---")
        print(f"Goal: Find '{goal}' | Givens: {[str(g) for g in givens]}")

        known_vars = set(givens)
        plan = []  # This will be populated by the recursive search

        # 1. Find the plan
        if not self._find_solution_path(givens, goal, plan, known_vars):
            print("Error: Could not find a solution path.\n")
            return

        print(f"Discovered Plan ({len(plan)} steps):")
        for i, step in enumerate(plan):
            print(
                f"  Step {i+1}: Solve for '{step['formula'].lhs}' using '{step['template_name']}'"
            )

        # 2. Execute the plan to get the answer

        # Pick random values for all "given" variables
        values = {}
        for sym in givens:
            values[str(sym)] = random.choice(self.all_pools[str(sym)])

        # Add misc values
        values["object_name"] = random.choice(self.all_pools["object_name"])

        # Calculate all bridge/final variables by executing the plan
        calculated_values = {}
        for step in plan:
            step_goal = str(step["formula"].lhs)
            # Get values needed for this step from both givens and prior steps
            input_vals = {
                str(k): (values.get(str(k)) or calculated_values.get(str(k)))
                for k in step["inputs"]
            }

            # Calculate and store the result
            result = step["formula"].rhs.subs(input_vals)
            calculated_values[step_goal] = result

        # The final answer is the last thing we calculated
        final_answer = calculated_values[str(goal)]

        # 3. Format the question using an LLM
        question_text = self._generate_problem_text_with_llm(
            givens, goal, values
        )  # No longer await

        if not question_text:
            print("Error: LLM failed to generate question text.")
            return

        # Get units for the final answer
        final_template = plan[-1]["template"]  # Template used for final step
        units = final_template["variables"][str(goal)].split("(")[-1].replace(")", "")

        print(f"Generated Q: {question_text}")
        print(f"Verifiable A: {final_answer} {units}")
        print("-" * 20)

    def _generate_problem_text_with_llm(self, givens, goal, values):  # No longer async
        """Helper function to call a local transformers LLM for problem text."""

        system_prompt = "You are a physics tutor. Your task is to create a single-paragraph word problem for a student. Do not include the answer, the formula, or the list of givens. Only output the final, natural-language word problem."

        # Build the prompt payload
        given_list_str = ""
        for sym in givens:
            val = values[str(sym)]
            desc = self.var_details.get(sym, str(sym))  # e.g., "mass (kg)"
            given_list_str += f"- {desc} = {val}\n"

        goal_desc = self.var_details.get(
            goal, str(goal)
        )  # e.g., "final velocity (m/s)"

        user_prompt = f"""
Please create a single-paragraph word problem that requires a student to solve for:
{goal_desc}

The problem must be solvable using the following given values:
{given_list_str}
- You can also invent a plausible object name like '{values['object_name']}'.

Combine these elements into a single, natural-language paragraph.
Do not list the givens. Do not state the answer.
"""
        # --- MODIFICATION: Use transformers pipeline ---
        try:
            if not self.llm_pipeline:
                print("Error: LLM pipeline was not initialized.")
                return None

            # Combine system and user prompt for the transformers pipeline
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Generate the text
            # We set max_new_tokens to get a reasonably long answer
            # and pad_token_id to suppress warnings for models like GPT-2
            result = self.llm_pipeline(
                full_prompt,
                max_new_tokens=100,  # Max tokens for the *answer*
                num_return_sequences=1,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
            )

            generated_text = result[0]["generated_text"]

            # The pipeline includes the prompt in the result, so we must remove it.
            # We find the generated part *after* the prompt.
            text = generated_text[len(full_prompt) :].strip()

            # Clean up the text
            if text:
                return text.strip().replace('"', "")
            else:
                print("Error: No text found in LLM response.")
                print(f"Full response: {result}")
                return None

        except Exception as e:
            print(f"An error occurred during the LLM pipeline call: {e}")
            return None
        # --- END MODIFICATION ---


# --- 5. Generator Functions (Now More Generic) ---


def generate_parametric_question(template, solve_for_symbol):
    """
    Method 1 & 2: Generates a question by parametrization.
    This function is now more robust and takes a symbol as an argument.
    """
    solve_for_var_str = str(solve_for_symbol)
    print(
        f"--- Generating (Method 1/2: Parametrization/Inversion) for '{solve_for_var_str}' ---"
    )

    try:
        # We only need the RHS of the solved equation
        solved_formula_rhs = solve(template["equation"], solve_for_symbol)[0]
    except IndexError:
        print(f"Error: Could not solve equation for {solve_for_var_str}")
        return

    given_vars = solved_formula_rhs.free_symbols
    values = {}
    for var in given_vars:
        var_name = str(var)
        if var_name in template["value_pools"]:
            values[var_name] = random.choice(template["value_pools"][var_name])
        else:
            print(f"Warning: No value pool for {var_name}")

    answer = solved_formula_rhs.subs(values)
    values[solve_for_var_str] = answer

    # Fill in the blanks in the text template
    question_text = template["text_templates"][solve_for_var_str].format(**values)
    units = template["variables"][solve_for_var_str].split("(")[-1].replace(")", "")

    print(f"Generated Q: {question_text}")
    print(f"Verifiable A: {answer} {units}")
    print("-" * 20)


# --- 6. Run the Generators ---
def main():  # No longer async
    """Synchronous main function to run our generators."""

    # Demo 1: Parametrization (The simplest case)
    # Goal: Solve for 'F'
    generate_parametric_question(force_template, F)

    # Demo 2: Formula Inversion
    # Goal: Solve for 'm'
    generate_parametric_question(force_template, m)

    # --- Demo 3: Dynamic Composition (Replaces the old hardcoded template) ---

    # 1. Initialize the solver. This builds the "brain".
    solver = ProblemSolver(TEMPLATE_DATABASE)

    # 2. Define our problem
    GIVEN_VARS = {F, m, v_i, t}
    GOAL_VAR = v_f

    # 3. Generate the question.
    #    The solver will automatically find the F,m -> a -> v_f path
    #    and call the LLM to generate the text.
    solver.generate_question(GIVEN_VARS, GOAL_VAR)  # No longer await


if __name__ == "__main__":
    # This is how you run a standard python script
    # NOTE: In a real environment, you need to install 'transformers' and 'torch'
    # pip install transformers torch
    main()
