import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Any # Added List, Tuple, Any for type hinting

# Import the base class
from .base_retriever import BaseRetriever

# Inherit from BaseRetriever
class EmbeddingRetriever(BaseRetriever):
    """
    Retrieves relevant text chunks based on semantic similarity using sentence embeddings.
    Inherits from BaseRetriever.
    """
    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", max_length=8192):
        """
        Initializes the EmbeddingRetriever.

        Args:
            model_name (str): The name of the Hugging Face model to use for embeddings.
            max_length (int): The maximum sequence length for the tokenizer.
        """
        super().__init__() # Call base class initializer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        # # Consider adding device management (e.g., self.device = 'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.device) # Move model to device
        # # and moving the model to the device: self.model.to(self.device)

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Helper function for pooling token embeddings."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
            ]

    # Implement the abstract method from BaseRetriever
    def vectorize_text(self, text_chunk: str) -> List[List[float]]:
        """
        Vectorizes a single text chunk into an embedding.

        Args:
            text_chunk (str): The text content to vectorize.

        Returns:
            List[List[float]]: The embedding vector, wrapped in a list
                               (consistent with potential batch processing,
                               although here it processes one chunk).
        """
        # Add device placement for batch_dict if device management is added in __init__
        # e.g., batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        batch_dict = self.tokenizer(
            [text_chunk], # Process as a list containing one item
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad(): # Inference mode
            outputs = self.model(**batch_dict)

        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        # Normalize and convert to list
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.tolist() # Returns List[List[float]]

    # Implement the abstract method from BaseRetriever
    def retrieve_relevant_chunks(
        self,
        query_representation: List[float], # Expecting a single query embedding vector
        document_representations: List[List[float]], # Expecting a list of document embedding vectors
        document_chunks_text: List[str],
        top_k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieves the most relevant document chunks based on cosine similarity
        between the query embedding and document embeddings.

        Note: This method calculates similarity in memory. For large datasets,
        using a vector database query (like ChromaDB's collection.query) is
        typically more efficient and is the approach used in rag_tester.py.
        This implementation fulfills the BaseRetriever contract but might not be
        directly used by rag_tester.py for embedding retrieval.

        Args:
            query_representation (List[float]): The embedding vector of the query.
            document_representations (List[List[float]]): A list of embedding vectors
                                                         for the documents.
            document_chunks_text (List[str]): The original text of the document chunks.
            top_k (int): The maximum number of relevant chunks to retrieve.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - A list of the most relevant document chunk texts.
                - A list of corresponding cosine similarity scores.
        """
        if not document_representations or not query_representation:
            return [], []

        # Ensure tensors are on the same device if using GPU
        # device = self.model.device # Or self.device if defined in __init__
        # query_embedding_tensor = torch.tensor(query_representation, device=device).unsqueeze(0) # Add batch dim
        # document_embeddings_tensor = torch.tensor(document_representations, device=device)

        # Simpler CPU version:
        query_embedding_tensor = torch.tensor(query_representation).unsqueeze(0) # Shape: (1, embed_dim)
        document_embeddings_tensor = torch.tensor(document_representations) # Shape: (num_docs, embed_dim)

        # Calculate cosine similarities (efficiently)
        # Normalize tensors just in case they aren't already (F.normalize was used in vectorize_text)
        query_embedding_tensor = F.normalize(query_embedding_tensor, p=2, dim=1)
        document_embeddings_tensor = F.normalize(document_embeddings_tensor, p=2, dim=1)

        # Cosine similarity = dot product of normalized vectors
        similarities = torch.matmul(document_embeddings_tensor, query_embedding_tensor.T).squeeze() # Shape: (num_docs,)

        # Get top_k results
        # Ensure top_k is not larger than the number of documents
        actual_top_k = min(top_k, len(document_chunks_text))
        if actual_top_k <= 0:
            return [], []

        # .topk returns values and indices
        top_k_similarities, top_k_indices = torch.topk(similarities, k=actual_top_k)

        # Convert indices and similarities to lists
        top_k_indices_list = top_k_indices.tolist()
        top_k_similarities_list = top_k_similarities.tolist()

        # Retrieve the corresponding text chunks
        relevant_chunks = [document_chunks_text[i] for i in top_k_indices_list]

        return relevant_chunks, top_k_similarities_list