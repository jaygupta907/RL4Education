from ppo_trainer import AgentLLM
import torch
from concept_graph import ConceptGraph
import yaml
from types import SimpleNamespace

def main(args):

    agent_llm = AgentLLM(args)

    question = "Explain the concept of overfitting in machine learning."

    agent_llm.ppo_step(question)

if __name__ == "__main__":
    args = yaml.load(open('config.yaml'),Loader=yaml.FullLoader)
    args = SimpleNamespace(**args)
    main(args)