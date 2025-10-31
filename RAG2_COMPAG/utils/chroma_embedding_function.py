# RAG2_COMPAG/utils/chroma_embedding_function.py
from chromadb import EmbeddingFunction, Documents, Embeddings
from typing import Any, List

class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB embedding function that wraps a pre-initialized
    EmbeddingRetriever instance.
    """
    def __init__(self, embedding_retriever: Any):
        """
        Initializes the function with an existing EmbeddingRetriever instance.

        Args:
            embedding_retriever: An initialized instance of EmbeddingRetriever
                                 (or any class with a `vectorize_document` method).
        """
        if not hasattr(embedding_retriever, 'vectorize_document'):
            raise TypeError("The provided embedding_retriever must have a 'vectorize_document' method.")
        self.embedding_retriever = embedding_retriever
        # It's useful to know the dimensionality of the embeddings
        self._dimensionality = self._get_embedding_dimensionality()

    def _get_embedding_dimensionality(self) -> int:
        """Helper to determine the output dimensionality of the model."""
        try:
            # Embed a dummy text and check the length of the resulting vector
            dummy_embedding = self.embedding_retriever.vectorize_document("test")
            if dummy_embedding and isinstance(dummy_embedding, list) and len(dummy_embedding) > 0:
                return len(dummy_embedding[0])
            else:
                raise ValueError("Could not determine embedding dimensionality from a test vectorization.")
        except Exception as e:
            print(f"Warning: Could not dynamically determine embedding dimensionality. Error: {e}")
            # Fallback to a common default or raise an error
            # Returning a placeholder; this might need adjustment depending on the models used.
            # A more robust solution might pass the dimension in __init__.
            return 4096 # Qwen3-Embedding-8B dimension


    def __call__(self, input: Documents) -> Embeddings:
        """
        Generates embeddings for a batch of documents.

        Args:
            input (Documents): A list of document strings.

        Returns:
            Embeddings: A list of embedding vectors.
        """
        all_embeddings: List[List[float]] = []
        for doc in input:
            # The vectorize_document method returns a List[List[float]], so we extract the inner list.
            embedding_list_of_lists = self.embedding_retriever.vectorize_document(doc)
            if embedding_list_of_lists and isinstance(embedding_list_of_lists, list) and len(embedding_list_of_lists) > 0:
                all_embeddings.append(embedding_list_of_lists[0])
            else:
                # If embedding fails, return a zero vector of the correct dimensionality
                # to avoid breaking ChromaDB operations, and log a warning.
                print(f"Warning: Embedding failed for document, returning zero vector. Document: {doc[:100]}...")
                all_embeddings.append([0.0] * self._dimensionality)
        return all_embeddings