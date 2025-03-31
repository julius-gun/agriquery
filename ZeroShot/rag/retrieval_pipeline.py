# RAG/retrieval_pipeline.py
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from pathlib import Path
from tempfile import mkdtemp

class RetrievalPipeline:
    """Handles the retrieval pipeline for RAG."""

    def __init__(self, rag_config):
        self.rag_config = rag_config
        self.retrieval_algorithm = rag_config.get("retrieval_algorithm", "keyword") # Default to keyword
        self.num_retrieved_documents = rag_config.get("num_retrieved_documents", 5)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") # Using HuggingFace embedding as in example
        self.embed_dim = len(self.embed_model.get_text_embedding("hi"))
        self.vector_store = MilvusVectorStore(
            uri=str(Path(mkdtemp()) / "docling.db"), # Using Milvus as in example
            dim=self.embed_dim,
            overwrite=True,
        )
        self.vector_store_index = None # Initialize index lazily

    def retrieve_relevant_context(self, question):
        """Retrieves relevant context based on the retrieval algorithm."""
        if self.retrieval_algorithm == "keyword":
            return self._keyword_retrieval(question)
        elif self.retrieval_algorithm == "semantic":
            return self._semantic_retrieval(question)
        elif self.retrieval_algorithm == "hybrid":
            return self._hybrid_retrieval(question)
        else:
            raise ValueError(f"Unknown retrieval algorithm: {self.retrieval_algorithm}")

    def _keyword_retrieval(self, question):
        """Placeholder for keyword-based retrieval."""
        print("Keyword retrieval is not yet implemented. Returning empty context.")
        return []

    def _semantic_retrieval(self, question):
        """Performs semantic retrieval using the vector store."""
        if self.vector_store_index is None:
            self.vector_store_index = self.vector_store.as_index(embed_model=self.embed_model) # Lazy index initialization
        query_engine = self.vector_store_index.as_query_engine(similarity_top_k=self.num_retrieved_documents)
        retrieval_results = query_engine.retrieve(question) # Use retrieve instead of query to get nodes
        return retrieval_results


    def _hybrid_retrieval(self, question):
        """Placeholder for hybrid retrieval."""
        print("Hybrid retrieval is not yet implemented. Returning semantic retrieval.")
        return self._semantic_retrieval(question) # Fallback to semantic for now