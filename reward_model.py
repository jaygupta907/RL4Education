import torch
import torch.nn as nn
import yaml
from sentence_transformers import SentenceTransformer
import networkx as nx
from transformers import pipeline
from concept_graph import ConceptGraph
import json

class RewardModel(nn.Module):
    def __init__(self, args):
        """
        Initialize the RewardModel class.

        Args:
            concept_graph_path (str): Path to the YAML file containing the concept graph.
            question_bank_path (list): List of questions to compute the mean question embedding.
        """
        super(RewardModel, self).__init__()
        # Load configuration arguments from 'config.yaml'
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.concept_graph_path = args.concept_graph_path
        self.question_bank_path = args.question_bank_path
        # Sentence embedding model for computing question embeddings
        self.embedding_model = SentenceTransformer('thenlper/gte-base').to(self.device)
        
        # Compute the mean embedding of the answers in the question bank
        self.mean_answer_embedding = self.compute_mean_answer_embedding(self.question_bank_path)
        
        # Zero-shot classification pipeline for concept extraction - use CPU to save GPU memory
        self.concept_extractor = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=-1)
        
        # Load the concept graph from the provided path
        self.concept_graph_class = ConceptGraph(self.concept_graph_path)
        self.concept_graph = self.concept_graph_class.get_concept_graph()
    
    def extract_concepts(self, question,threshold=0.95):
        """
        Extract relevant concepts from the question using zero-shot classification.

        Args:
            question (str): The input question.
            concepts (list, optional): List of concepts to classify against. Defaults to None.
            threshold (float): Minimum score to consider a concept as relevant.

        Returns:
            list: Relevant concepts extracted from the question.
        """
        # Use all nodes in the concept graph as potential concepts
        concepts = self.concept_graph_class.get_concepts()
   
        # Perform zero-shot classification
        results = self.concept_extractor(question, concepts, multi_label=True)

        # Filter concepts based on the threshold
        relevant_concepts = []
        for label, score in zip(results['labels'], results['scores']):
            if score >= threshold:
                relevant_concepts.append(label)

        return relevant_concepts
    
    def compute_mean_answer_embedding(self, question_bank_path):
            """
            Loads answers from a JSON file and computes their mean embedding.

            Args:
                question_bank_path (str): The file path to the JSON question bank.

            Returns:
                torch.Tensor: The mean embedding of all answers in the bank.
            """
            # 1. Load the JSON and extract all answer strings into a list
            answers = []
            with open(question_bank_path, 'r') as f:
                data = json.load(f)

            for concept_block in data:
                if 'questions' in concept_block and isinstance(concept_block['questions'], list):
                    for qa_pair in concept_block['questions']:
                        if 'answer' in qa_pair:
                            answers.append(qa_pair['answer'])

            if not answers:
                print("No answers found in the file.")
                return None

            # 2. Encode the list of answers
            embeddings = self.embedding_model.encode(answers, convert_to_tensor=True)

            # 3. Compute the mean of the embeddings
            # The embeddings are already a tensor, so no need for torch.tensor()
            mean_embedding = torch.mean(embeddings, dim=0)

            return mean_embedding

    def novelity_reward(self, answer_text):
        """
        Compute the novelty reward for an answer based on its embedding similarity
        to the mean answer embedding.

        Args:
            answer_text (str): The generated answer text.

        Returns:
            float: Novelty reward (1 - cosine similarity).
        """
        if self.mean_answer_embedding is None:
            return 0.0
        answer_embedding = self.embedding_model.encode([answer_text], convert_to_tensor=True)
        novelty = torch.cosine_similarity(torch.tensor(answer_embedding), self.mean_answer_embedding.unsqueeze(0))
        return 1.0 - novelty.item()
    
    def concept_depth_reward(self, question):
        """
        Compute the concept depth reward based on the depth of extracted concepts
        in the concept graph.

        Args:
            question (str): The input question.

        Returns:
            int: Maximum depth of the extracted concepts in the graph.
        """
        # Identify root nodes (nodes with no prerequisites)
        root_nodes = [node for node in self.concept_graph.nodes() if self.concept_graph.in_degree(node) == 0]
        
        # Extract relevant concepts from the question
        extracted_concepts = self.extract_concepts(question)

        outer_max_list = []
        for q_concept in extracted_concepts:
            if q_concept not in self.concept_graph:
                continue

            distances_to_this_concept = []
            for root in root_nodes:
                try:
                    # Compute shortest path length from root to the concept
                    distance = nx.shortest_path_length(self.concept_graph, source=root, target=q_concept)
                    distances_to_this_concept.append(distance)
                except nx.NetworkXNoPath:
                    continue
            
            if distances_to_this_concept:
                inner_max_dist = max(distances_to_this_concept)
                outer_max_list.append(inner_max_dist)

        if not outer_max_list:
            return 0  # Return 0 if no valid concepts are found

        return max(outer_max_list)

    def get_reward(self, question, allowed_concepts=None, answer=None):
        """
        Compute the total reward for a question based on novelty and depth rewards.

        Args:
            question (str): The input question.

        Returns:
            float: Total reward for the question.
        """

        novelty = self.novelity_reward(answer) if isinstance(answer, str) and len(answer.strip()) > 0 else 0.0
        depth = self.concept_depth_reward(question)
        
        # Weighted sum of individual rewards
        total_reward = self.args.novelty * novelty + self.args.depth * depth
        return total_reward