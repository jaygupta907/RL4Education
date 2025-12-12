import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch

class LoadContext:
    """Loads and formats context documents for a given question using a retriever."""

    def __init__(self, question_bank_path, k=5):
        """
        Initializes the LoadContext class.

        Args:
            question_bank_path (str): Path to the JSON file containing the question bank.
            k (int): Number of top documents to retrieve for context.
        """
        # Load the question-answer pairs from the JSON file
        self.docs = self.load_qa_from_json(question_bank_path)
        
        # Create a retriever using FAISS for document retrieval
        self.retriever = self.create_retriever(self.docs, k)
        
        # Define the prompt template for the LLM
        self.template = """
        You are an AI assistant for answering questions about machine learning.
        Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Provide a detailed and well-explained answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        self.prompt = PromptTemplate.from_template(self.template)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the HuggingFace pipeline for text generation - use CPU to save GPU memory
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 150},
            device=-1  # Use CPU instead of GPU
        )
        
        # Define the RAG (Retrieval-Augmented Generation) chain
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def load_qa_from_json(self, file_path):
        """
        Loads the Q&A data and creates LangChain Documents.

        Args:
            file_path (str): Path to the JSON file containing the question bank.

        Returns:
            list: A list of LangChain Document objects.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        documents = []
        for concept_block in data:
            # Ensure the 'questions' field exists and is a list
            if 'questions' in concept_block and isinstance(concept_block['questions'], list):
                for qa_pair in concept_block['questions']:
                    # Only include Q&A pairs with both 'question' and 'answer'
                    if 'question' in qa_pair and 'answer' in qa_pair:
                        documents.append(
                            Document(
                                page_content=qa_pair['question'],
                                metadata={
                                    'concept': concept_block.get('concept', 'Unknown'),
                                    'answer': qa_pair['answer']
                                }
                            )
                        )
        return documents
    
    def create_retriever(self, documents, k):
        """
        Creates a FAISS vector store and retriever from the documents.

        Args:
            documents (list): List of LangChain Document objects.
            k (int): Number of top documents to retrieve.

        Returns:
            FAISS retriever: A retriever object for retrieving relevant documents.
        """
        # Initialize the embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create a FAISS vector store from the documents
        vector_store = FAISS.from_documents(documents, embedding_model)
        
        # Return the retriever with the specified number of top results
        return vector_store.as_retriever(search_kwargs={"k": k})

    def format_docs(self, docs):
        """
        Formats the retrieved documents into a readable string for the LLM.

        Args:
            docs (list): List of retrieved LangChain Document objects.

        Returns:
            str: A formatted string containing the retrieved questions and answers.
        """
        return "\n\n".join(
            f"Retrieved Question: {doc.page_content}\nRetrieved Answer: {doc.metadata['answer']}"
            for doc in docs
        )

    def get_context(self, topic: str) -> str:
        """
        Retrieves and formats context documents for the given question.

        Args:
            question (str): The input question.

        Returns:
            tuple: A tuple containing the formatted context and the LLM's response.
        """
        # Retrieve relevant documents for the question
        retrieved_docs = self.retriever.invoke(topic)
        
        # Format the retrieved documents into a readable context
        formatted_context = self.format_docs(retrieved_docs)
        return formatted_context
    
    def get_response(self, question: str) -> str:
        """
        Generates a response for the given question using the RAG chain.

        Args:
            question (str): The input question.
        Returns:
            str: The generated response from the LLM.
        """
        return self.rag_chain.invoke(question)