from ppo_trainer import AgentLLM
import torch
from concept_graph import ConceptGraph
import yaml
from types import SimpleNamespace

def main(args):

    agent_llm = AgentLLM(args)

    # Initial question(s)
    questions = ["How does Bias-Variance Tradeoff affect error during training."]

    # Loop to iteratively refine responses using PPO
    for _ in range(args.num_iterations):  # Number of iterations specified in config
        all_decoded_responses = []  # To store responses from all questions in this iteration

        for question in questions:
            # Run PPO step for each question
            decoded_responses, rewards, stats = agent_llm.ppo_step(question)
            all_decoded_responses.extend(decoded_responses)  # Collect all responses


        # Use all responses as questions for the next iteration
        questions = all_decoded_responses
        

if __name__ == "__main__":
    args = yaml.load(open('config.yaml'),Loader=yaml.FullLoader)
    args = SimpleNamespace(**args)
    main(args)