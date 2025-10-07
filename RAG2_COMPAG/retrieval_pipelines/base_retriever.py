# RAG/retrieval_pipelines/base_retriever.py
from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class BaseRetriever(ABC):
    """
    Abstract base class for all retriever implementations.

    Defines the common interface that retrieval pipelines should adhere to,
    allowing for interchangeable retrieval strategies (e.g., embedding, keyword).
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initializes the retriever. Specific parameters depend on the implementation.
        """
        pass

    @abstractmethod
    def vectorize_text(self, text_chunk: str) -> Any:
        """
        Processes a text chunk into a representation suitable for the specific
        retrieval method.

        For embedding retrievers, this typically returns a numerical vector (e.g., List[float]).
        For keyword retrievers, this might return processed text, keywords, or another format.

        Args:
            text_chunk (str): The text content to process.

        Returns:
            Any: The processed representation of the text chunk.
        """
        pass

    @abstractmethod
    def retrieve_relevant_chunks(
        self,
        query_representation: Any,
        document_representations: Any, # This might be embeddings, indices, or other structures
        document_chunks_text: List[str],
        top_k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieves the most relevant document chunks based on the query representation.

        Note: The exact mechanism and the nature of 'document_representations'
        will vary significantly between retriever types. For example, embedding
        retrievers might compare against document embeddings, while keyword
        retrievers might use an inverted index or TF-IDF scores.
        The current implementation in `rag_tester.py` handles ChromaDB queries
        separately for embedding retrieval, bypassing a direct call to this method
        on the `EmbeddingRetriever` instance after vectorization. This base method
        provides a more general interface signature.

        Args:
            query_representation (Any): The processed representation of the user query
                                        (output of `vectorize_text` or similar).
            document_representations (Any): A structure containing the representations
                                            of the documents to search within (e.g.,
                                            a list of embeddings, a search index).
            document_chunks_text (List[str]): The original text of the document chunks,
                                              indexed consistently with document_representations.
            top_k (int): The maximum number of relevant chunks to retrieve.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - A list of the most relevant document chunk texts.
                - A list of corresponding relevance scores (e.g., similarity scores).
                  If scoring is not applicable, a list of default values (e.g., 0.0)
                  can be returned.
        """
        pass