import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class EmbeddingRetriever: # Renamed from KeywordRetriever to EmbeddingRetriever
    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", max_length=8192):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
            ]

    def vectorize_text(self, text_chunk):
        batch_dict = self.tokenizer(
            [text_chunk],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # print(f"Input Chunk (first 50 chars): {text_chunk[:50]}...") # Print start of chunk
        # print(f"Input Token Length: {batch_dict['input_ids'].shape[1]}") # Print token length of input

        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1).tolist()

    def retrieve_relevant_chunks(self, question_embedding, document_chunk_embeddings, document_chunks_text, top_k=3):
        question_embedding_tensor = torch.tensor(question_embedding)
        document_chunk_embeddings_tensor = [torch.tensor(emb) for emb in document_chunk_embeddings]
        similarities = [F.cosine_similarity(question_embedding_tensor, chunk_embedding).item() for chunk_embedding in document_chunk_embeddings_tensor]
        best_chunk_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        relevant_chunks = [document_chunks_text[i] for i in best_chunk_indices] # Return text chunks instead of indices
        return relevant_chunks, [similarities[i] for i in best_chunk_indices] # Return chunks and similarities