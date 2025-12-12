from question_judge import QuestionJudge
from tree_walk_calculation import TreeWalkCalculator
from generate_question_from_answer import QuestionGenerator

calculator = TreeWalkCalculator("variable_concept_graph.json", max_length=4)
result = calculator.run("magnetic_flux", min_val=1.0, max_val=100.0)

generator = QuestionGenerator()
question = generator.generate_question(calculator)

judge = QuestionJudge()
score = judge.evaluate(calculator, question)  # Returns 0.0-10.0
detailed = judge.evaluate_detailed(calculator, question)  # Returns dict with score + feedback

print(f"Score: {score}")
print(f"Detailed: {detailed}")