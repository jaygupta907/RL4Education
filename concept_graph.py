import networkx as nx
import json
import random
import time
import os

class ConceptGraph:
    def __init__(self, concept_graph_path):
        """
        Initialize the ConceptGraph class.

        Args:
            concept_graph_path (str): Path to the YAML file containing the concept graph.
        """
        self.concept_graph_path = concept_graph_path
        self.concept_graph = self.get_concept_graph()

    def get_concept_graph(self):
            """
            Load the concept graph from a JSON file and construct a directed graph.

            Returns:
                nx.DiGraph: A directed graph representing concepts and their prerequisites.
            """
            G = nx.DiGraph()
            
            # Open and load the JSON file
            with open(self.concept_graph_path, 'r') as f:
                # --- This is the corrected line ---
                concepts = json.load(f)
            
            # Add nodes and edges to the graph
            for item in concepts:
                # Ensure the 'concept' key exists before adding
                if 'concept' in item:
                    concept = item['concept']
                    G.add_node(concept)
                    
                    # Ensure 'prerequisites' key exists and is a list
                    if 'prerequisites' in item and isinstance(item['prerequisites'], list):
                        for prereq in item['prerequisites']:
                            # The has_node check is redundant as add_node does nothing if the node exists,
                            # but it's harmless to keep.
                            G.add_node(prereq)
                            G.add_edge(prereq, concept) # Edge from prerequisite -> concept
            return G
    
    def get_concepts(self):
        """
        Get the list of concepts in the graph.

        Returns:
            list: A list of concept names.
        """
        return list(self.concept_graph.nodes)
    
    def get_dependents(self, concepts):
        """
        Get the dependents for a given concept.

        Args:
            concept (str): The concept for which to retrieve dependents.
        """
        dependents = set()
        for concept in concepts:
            if concept in self.concept_graph:
                successors = list(self.concept_graph.successors(concept))
                dependents.update(successors)
        return list(dependents)
    
    def get_prerequisites(self, concept):
        """
        Get the prerequisites for a given concept.

        Args:
            concept (str): The concept for which to retrieve prerequisites.
        """
        if concept in self.concept_graph:
            return list(self.concept_graph.predecessors(concept))
        else:
            return []
    
    # def validate_walk(self, walk):
    #     """
    #     Validate that a walk follows valid prerequisite relationships.
        
    #     Args:
    #         walk (list): A list of concepts representing a walk.
        
    #     Returns:
    #         tuple: (is_valid, invalid_edges) where is_valid is bool and invalid_edges is list of tuples
    #     """
    #     if len(walk) < 2:
    #         return True, []
        
    #     invalid_edges = []
    #     for i in range(len(walk) - 1):
    #         current = walk[i]
    #         next_node = walk[i + 1]
            
    #         # Check if there's an edge from current to next_node
    #         # This means current must be a prerequisite of next_node
    #         if not self.concept_graph.has_edge(current, next_node):
    #             invalid_edges.append((current, next_node))
        
    #     return len(invalid_edges) == 0, invalid_edges
    
    def random_walk(self, length, start_node=None):
        """
        Perform a directed random walk of fixed length on the concept graph.
        Starts from a node and picks any neighbor (successor) randomly and unidirectionally.
        If length is less than required and a leaf node (no successors) is reached, returns the path traversed.
        
        Args:
            length (int): The desired length of the random walk.
            start_node (str, optional): Starting node. If None, starts from a random node.
        
        Returns:
            list: A list of nodes visited during the random walk.
        """
        # Add entropy to ensure different walks across runs
        # Mix time-based seed with OS random for better randomization
        entropy = int(time.time() * 1000000) ^ int.from_bytes(os.urandom(4), 'big')
        random.seed(entropy % (2**32))
        
        if start_node is None:
            # Start from a random node
            nodes = list(self.concept_graph.nodes())
            if not nodes:
                return []
            # Shuffle for extra randomness
            random.shuffle(nodes)
            start_node = random.choice(nodes)
        
        walk = [start_node]
        current = start_node
        
        for _ in range(length-1):
            # Get successors (neighbors in forward direction)
            # Successors are concepts that have 'current' as a prerequisite
            # This means we can only go from a prerequisite to concepts that depend on it
            successors = list(self.concept_graph.successors(current))
            
            # If no successors, we've reached a leaf node - return path traversed so far
            if not successors:
                break
            
            # Shuffle successors for extra randomness before picking
            random.shuffle(successors)
            # Pick a random neighbor (successor)
            # This will be a concept that has 'current' as a prerequisite
            next_node = random.choice(successors)
            
            current = next_node
            walk.append(current)
        
        return walk