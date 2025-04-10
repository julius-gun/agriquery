# RAG/retrieval_pipelines/keyword_retrieval.py
from typing import List, Any, Tuple, Optional # Added Optional
import logging # Import logging

# Import the base class
from .base_retriever import BaseRetriever

# Attempt to import rank_bm25, provide guidance if missing
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logging.error("The 'rank_bm25' library is required for KeywordRetriever. Please install it using: pip install rank-bm25")
    # Option 1: Raise the error immediately
    # raise ImportError("The 'rank_bm25' library is required for KeywordRetriever. Please install it using: pip install rank-bm25")
    # Option 2: Define a dummy class or allow initialization but fail later (less ideal)
    BM25Okapi = None # Set to None so checks later fail gracefully if not installed

# Inherit from BaseRetriever
class KeywordRetriever(BaseRetriever):
    """
    Retrieves relevant text chunks based on keyword matching using the BM25 algorithm.
    Inherits from BaseRetriever.

    Requires the 'rank_bm25' library to be installed.
    """
    def __init__(self, document_chunks: Optional[List[str]] = None):
        """
        Initializes the KeywordRetriever and builds the BM25 index.

        Args:
            document_chunks (Optional[List[str]]): A list of text chunks representing the corpus
                                                  to search over. If None, the retriever is
                                                  initialized but cannot perform retrieval until
                                                  an index is built using `build_index`.
        """
        super().__init__() # Call base class initializer
        self.corpus: Optional[List[str]] = None
        self.tokenized_corpus: Optional[List[List[str]]] = None
        self.bm25: Optional[BM25Okapi] = None

        if BM25Okapi is None:
             logging.warning("KeywordRetriever initialized WITHOUT BM25 capabilities because 'rank_bm25' is not installed.")
             # Or raise an error here if BM25 is strictly required at init time
             # raise RuntimeError("Cannot initialize KeywordRetriever: 'rank_bm25' library not found.")
        elif document_chunks:
            self.build_index(document_chunks)
        else:
            logging.info("KeywordRetriever initialized without documents. Call 'build_index' before retrieval.")


    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer and lowercasing.
        Can be replaced with a more sophisticated tokenizer if needed.
        """
        # Consider adding stop word removal or stemming for better results
        return text.lower().split()

    def build_index(self, document_chunks: List[str]):
        """
        Builds or rebuilds the BM25 index from the provided document chunks.

        Args:
            document_chunks (List[str]): The corpus of text chunks.
        """
        if BM25Okapi is None:
            logging.error("Cannot build index: 'rank_bm25' library not found.")
            return # Or raise error

        if not document_chunks:
            logging.warning("Attempted to build index with empty document list.")
            self.corpus = []
            self.tokenized_corpus = []
            self.bm25 = None
            return

        logging.info(f"Building BM25 index for {len(document_chunks)} document chunks...")
        self.corpus = document_chunks
        # Tokenize the corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        # Create the BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logging.info("BM25 index built successfully.")

    # Implement the abstract method from BaseRetriever
    def vectorize_text(self, text_chunk: str) -> List[str]:
        """
        Processes a text chunk (query) for keyword retrieval by tokenizing it.

        Args:
            text_chunk (str): The query text to process.

        Returns:
            List[str]: The tokenized query.
        """
        # For BM25, the "vectorization" of the query is just tokenization
        tokenized_query = self._tokenize(text_chunk)
        # logging.debug(f"KeywordRetriever: Tokenized query: {tokenized_query}") # Optional debug
        return tokenized_query

    # Implement the abstract method from BaseRetriever
    def retrieve_relevant_chunks(
        self,
        query_representation: List[str], # Expecting tokenized query from vectorize_text
        document_representations: Any = None, # Not directly used, index is internal
        document_chunks_text: Optional[List[str]] = None, # Can be used for verification or if index not built
        top_k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieves relevant chunks based on BM25 scores.

        Args:
            query_representation (List[str]): The tokenized query.
            document_representations (Any): Not used by this BM25 implementation.
            document_chunks_text (Optional[List[str]]): The original text chunks. If provided,
                                                        it's used to return the results. If None,
                                                        the internal corpus stored during init/build_index
                                                        is used.
            top_k (int): The maximum number of relevant chunks to retrieve.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - A list of the most relevant document chunk texts.
                - A list of corresponding BM25 scores.

        Raises:
            ValueError: If the BM25 index has not been built before calling retrieve.
            RuntimeError: If the 'rank_bm25' library was not installed.
        """
        if self.bm25 is None:
            if BM25Okapi is None:
                 raise RuntimeError("Cannot retrieve: 'rank_bm25' library not found.")
            else:
                 raise ValueError("BM25 index not built. Call 'build_index' with document chunks first.")

        # Use the internal corpus if document_chunks_text is not provided
        corpus_to_use = document_chunks_text if document_chunks_text is not None else self.corpus
        if corpus_to_use is None:
             raise ValueError("No document corpus available for retrieval.")
        if len(corpus_to_use) != len(self.tokenized_corpus):
             logging.warning("Mismatch between provided document_chunks_text and indexed corpus size. Using indexed corpus.")
             corpus_to_use = self.corpus # Fallback to internal corpus

        # Get BM25 scores for the query against all documents in the index
        # Note: query_representation is already the tokenized query from vectorize_text
        scores = self.bm25.get_scores(query_representation)

        # Get the indices of the top_k scores
        # Ensure top_k is not larger than the number of documents
        actual_top_k = min(top_k, len(corpus_to_use))
        if actual_top_k <= 0:
            return [], []

        # Get indices of top scores (argsort gives indices of smallest first, so we reverse)
        # Alternatively, use a method that directly gives top N indices if available,
        # or partition/sort for efficiency on very large corpora.
        # Simple approach: get all scores, sort indices by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_k_indices = sorted_indices[:actual_top_k]

        # Retrieve the corresponding text chunks and their scores
        relevant_chunks = [corpus_to_use[i] for i in top_k_indices]
        top_scores = [scores[i] for i in top_k_indices]

        # logging.debug(f"KeywordRetriever: Retrieved {len(relevant_chunks)} chunks with scores: {top_scores}") # Optional debug
        return relevant_chunks, top_scores