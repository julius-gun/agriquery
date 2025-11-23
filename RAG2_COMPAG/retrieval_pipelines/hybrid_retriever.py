import logging
from typing import List, Dict, Any, Tuple, Optional

# Import base class and constituent retrievers
from .base_retriever import BaseRetriever
from .embedding_retriever import EmbeddingRetriever
from .keyword_retrieval import KeywordRetriever # Assuming BM25Okapi is available

# Import ChromaDB client type hint if needed for direct interaction
import chromadb

# Constants for RRF (Reciprocal Rank Fusion)
RRF_K = 60 # Common value for RRF constant

class HybridRetriever(BaseRetriever):
    """
    Retrieves relevant text chunks by combining results from both
    embedding-based semantic search and keyword-based search (BM25).

    Uses Reciprocal Rank Fusion (RRF) to combine the ranked lists.
    """

    def __init__(
        self,
        embedding_retriever: Optional[EmbeddingRetriever] = None, # Accepts pre-initialized retriever
        embedding_model_config: Optional[Dict[str, Any]] = None, # Fallback to create a new one
        chroma_client: Optional[chromadb.ClientAPI] = None,
        collection_name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initializes the HybridRetriever.

        Prioritizes using a pre-initialized embedding_retriever if provided.
        Otherwise, it creates a new one using embedding_model_config.

        Args:
            embedding_retriever (Optional[EmbeddingRetriever]): A pre-initialized instance of EmbeddingRetriever.
            embedding_model_config (Optional[Dict[str, Any]]): Configuration for the embedding model (used as fallback).
            chroma_client (Optional[chromadb.ClientAPI]): Initialized ChromaDB client. Required for embedding search.
            collection_name (Optional[str]): Name of the ChromaDB collection to query. Required for embedding search.
            verbose (bool): If True, enables detailed logging during retrieval.
        """
        super().__init__()
        self.verbose = verbose
        logging.info(f"Initializing HybridRetriever...")

        # --- Embedding Component ---
        if embedding_retriever:
            logging.info("  Using shared embedding retriever instance for HybridRetriever.")
            self.embedding_retriever = embedding_retriever
        elif embedding_model_config:
            logging.info("  Creating new embedding retriever instance for HybridRetriever (fallback).")
            self.embedding_retriever = EmbeddingRetriever(
                model_config=embedding_model_config
            )
        else:
            raise ValueError("HybridRetriever requires either an 'embedding_retriever' instance or an 'embedding_model_config'.")

        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.collection: Optional[chromadb.Collection] = None # Will be fetched later

        if self.chroma_client and self.collection_name:
             try:
                 # TODO: Handle embedding function requirement if needed by get_collection
                 self.collection = self.chroma_client.get_collection(name=self.collection_name)
                 logging.info(f"HybridRetriever: Successfully connected to Chroma collection '{self.collection_name}'")
             except Exception as e:
                 logging.error(f"HybridRetriever: Failed to get Chroma collection '{self.collection_name}': {e}. Embedding search will fail.")
                 # Decide if this should be a fatal error
                 # raise ValueError(f"HybridRetriever requires a valid Chroma client and collection name.") from e
        else:
             logging.warning("HybridRetriever initialized without Chroma client or collection name. Embedding search will not be possible.")


        # --- Keyword Component ---
        # Initialize KeywordRetriever without documents initially.
        # Index will be built later via `build_keyword_index`.
        self.keyword_retriever = KeywordRetriever(document_chunks=None)

        logging.info(f"HybridRetriever initialized.")


    def build_keyword_index(self, document_chunks: List[str]):
        """
        Builds the keyword (BM25) index using the provided document chunks.
        Delegates to the internal KeywordRetriever.

        Args:
            document_chunks (List[str]): The corpus of text chunks.
        """
        if not document_chunks:
             logging.warning("HybridRetriever: Attempting to build keyword index with empty document list.")
             # Ensure the internal retriever handles this gracefully
             self.keyword_retriever.build_index([])
             return

        logging.info(f"HybridRetriever: Building keyword index for {len(document_chunks)} documents...")
        try:
            self.keyword_retriever.build_index(document_chunks)
            logging.info(f"HybridRetriever: Keyword index built successfully.")
        except Exception as e:
            logging.error(f"HybridRetriever: Error building keyword index: {e}", exc_info=True)
            # Decide how to handle this - maybe raise?
            raise


    # Implement the abstract method from BaseRetriever
    def vectorize_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a query string into representations suitable for
        both embedding and keyword retrieval.

        Args:
            query (str): The query text to process.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'embedding': The embedding vector (List[float]).
                - 'tokens': The tokenized query (List[str]).
        """
        if self.verbose: logging.info(f"HybridRetriever: Vectorizing query: '{query[:50]}...'")

        # Get embedding (returns List[List[float]], take the first)
        embedding_vector = self.embedding_retriever.vectorize_query(query)[0]

        # Get keyword tokens
        keyword_tokens = self.keyword_retriever.vectorize_query(query)

        return {
            "embedding": embedding_vector,
            "tokens": keyword_tokens
        }

    # Implement the abstract method from BaseRetriever
    def vectorize_document(self, document: str) -> List[List[float]]:
        """
        Processes a document string for embedding. This is a pass-through
        to the internal embedding retriever.

        Args:
            document (str): The document text to process.

        Returns:
            List[List[float]]: The embedding of the document.
        """
        return self.embedding_retriever.vectorize_document(document)


    # Implement the abstract method from BaseRetriever
    def retrieve_relevant_chunks(
        self,
        query_representation: Dict[str, Any], # Expecting output from vectorize_query
        document_representations: Any = None, # Not directly used here, retrievers manage their own data
        document_chunks_text: Optional[List[str]] = None, # Keyword retriever might use its internal corpus
        top_k: int = 5 # Number of results desired *after* fusion
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieves relevant chunks using both embedding and keyword search,
        then combines the results using Reciprocal Rank Fusion (RRF).

        Args:
            query_representation (Dict[str, Any]): The processed query containing
                                                   'embedding' and 'tokens'.
            document_representations (Any): Not directly used by this implementation.
            document_chunks_text (Optional[List[str]]): Optional list of original texts.
                                                        If None, keyword retriever uses its internal corpus.
            top_k (int): The desired number of final results after fusion. Note that
                         more results might be fetched internally from each retriever.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - A list of the most relevant document chunk texts, ranked by RRF.
                - A list of corresponding RRF scores.

        Raises:
            ValueError: If components (Chroma collection, Keyword index) are missing.
            RuntimeError: If 'rank_bm25' library is missing.
        """
        if self.verbose: logging.info(f"HybridRetriever: Starting retrieval for top_k={top_k}")

        # --- Input Validation ---
        if 'embedding' not in query_representation or 'tokens' not in query_representation:
            raise ValueError("Invalid query_representation. Must contain 'embedding' and 'tokens'.")

        # --- Parameters ---
        # Fetch more results initially to allow for better fusion
        # Heuristic: fetch 2*top_k from each, capped reasonably
        fetch_k = max(top_k * 2, 10)

        # --- 1. Embedding Search (ChromaDB) ---
        embedding_results_docs = []
        embedding_results_ids = []
        embedding_scores = [] # ChromaDB query returns distances/similarities
        if self.collection:
            try:
                # Check if collection is empty to avoid "Nothing found on disk" HNSW errors
                collection_count = self.collection.count()
                
                if collection_count == 0:
                     if self.verbose: logging.warning(f"HybridRetriever: Collection '{self.collection_name}' is empty. Skipping embedding search.")
                else:
                    if self.verbose: logging.info(f"HybridRetriever: Performing embedding search (fetch_k={fetch_k})...")
                    query_embedding = [query_representation['embedding']] # Needs to be List[List[float]]
                    
                    # Ensure we don't ask for more results than exist, though Chroma usually handles it, 
                    # explicitly setting it prevents edge cases.
                    effective_k = min(fetch_k, collection_count)
                    
                    results = self.collection.query(
                        query_embeddings=query_embedding,
                        n_results=effective_k,
                        include=['documents', 'distances'] # Or 'similarities' depending on space
                    )
                    # Process results (handle potential Nones or empty lists)
                    if results and results.get('ids') and results['ids'][0]:
                        embedding_results_ids = results['ids'][0]
                        embedding_results_docs = results['documents'][0] if results.get('documents') else [""] * len(embedding_results_ids) # Fetch docs if available
                        # Chroma returns distances by default (lower is better). We need ranks.
                        if self.verbose: logging.info(f"HybridRetriever: Embedding search found {len(embedding_results_ids)} results.")
                    else:
                        if self.verbose: logging.info("HybridRetriever: Embedding search returned no results.")

            except Exception as e:
                logging.error(f"HybridRetriever: Error during embedding search: {e}", exc_info=True)
                # Continue without embedding results.
        else:
            logging.warning("HybridRetriever: Skipping embedding search (Chroma collection not available).")


        # --- 2. Keyword Search (BM25) ---
        keyword_results_docs = []
        keyword_scores = [] # BM25 scores (higher is better)
        try:
            if self.verbose: logging.info(f"HybridRetriever: Performing keyword search (fetch_k={fetch_k})...")
            # KeywordRetriever.retrieve_relevant_chunks expects tokenized query
            tokenized_query = query_representation['tokens']
            # It uses its internal corpus unless document_chunks_text is provided
            keyword_results_docs, keyword_scores = self.keyword_retriever.retrieve_relevant_chunks(
                query_representation=tokenized_query,
                top_k=fetch_k,
                document_chunks_text=document_chunks_text # Pass along if provided
            )
            if self.verbose: logging.info(f"HybridRetriever: Keyword search found {len(keyword_results_docs)} results.")
        except (ValueError, RuntimeError) as e: # Catch index not built or library missing
            logging.error(f"HybridRetriever: Error during keyword search: {e}", exc_info=True)
            # Continue without keyword results? Or raise? Let's continue.
        except Exception as e:
            logging.error(f"HybridRetriever: Unexpected error during keyword search: {e}", exc_info=True)
            # Continue without keyword results? Or raise? Let's continue.


        # --- 3. Combine Results (Reciprocal Rank Fusion - RRF) ---
        if self.verbose: logging.info(f"HybridRetriever: Combining results using RRF (k={RRF_K})...")

        # Need a way to map results to unique documents/chunks.
        # Using the text content itself as the key can be problematic if chunks are identical.
        # If ChromaDB IDs correspond reliably to the order/content used for BM25 indexing, we could use IDs.
        # Let's assume the `document_chunks_text` provided (or the internal corpus of KeywordRetriever)
        # matches the documents stored in ChromaDB. We'll use the *text* as the key for fusion.
        # This requires having the document text from both retrievers.

        # Create ranked lists (doc_text -> rank)
        # Rank starts at 1
        embedding_ranks = {doc: rank + 1 for rank, doc in enumerate(embedding_results_docs)}
        keyword_ranks = {doc: rank + 1 for rank, doc in enumerate(keyword_results_docs)}

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        all_docs = set(embedding_ranks.keys()) | set(keyword_ranks.keys())

        for doc in all_docs:
            score = 0.0
            if doc in embedding_ranks:
                score += 1.0 / (RRF_K + embedding_ranks[doc])
            if doc in keyword_ranks:
                score += 1.0 / (RRF_K + keyword_ranks[doc])
            rrf_scores[doc] = score

        # Sort documents by RRF score (descending)
        sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        # Get the top_k results
        final_docs = sorted_docs[:top_k]
        final_scores = [rrf_scores[doc] for doc in final_docs]

        if self.verbose: logging.info(f"HybridRetriever: RRF resulted in {len(final_docs)} final documents.")

        return final_docs, final_scores
