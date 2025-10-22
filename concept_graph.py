import networkx as nx
import json

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