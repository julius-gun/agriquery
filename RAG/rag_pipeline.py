import chromadb
from chromadb.utils import embedding_functions
import hashlib
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
from evaluation.evaluation_metrics import evaluate_rag_pipeline
from analysis.analysis_tools import (
    analyze_evaluation_results,
    load_dataset,
    analyze_dataset_across_types,
)
import json
import os

# NEW IMPORT
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
persist_directory = "chroma_db"
collection_name = "english_manual"
path_to_text = "english_manual.txt"

# --- Dataset paths are now defined here, can be moved to a config file later ---
dataset_paths = {
    "general_questions": "question_datasets/question_answers_pairs.json",
    "table_questions": "question_datasets/question_answers_tables.json",
    "unanswerable_questions": "question_datasets/question_answers_unanswerable.json",
}
config_file_path = "config.json"  # Path to config file

def initialize_retriever(retrieval_strategy_str: str):
    """Initializes the retriever based on the specified strategy."""
    if retrieval_strategy_str == "embedding":
        # Ensure EmbeddingRetriever is imported
        from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
        return EmbeddingRetriever()
    elif retrieval_strategy_str == "keyword":
        # Ensure KeywordRetriever is imported
        from retrieval_pipelines.keyword_retrieval import KeywordRetriever
        return KeywordRetriever()
    elif retrieval_strategy_str == "hybrid":
        raise NotImplementedError("Hybrid retrieval not yet implemented.")
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy_str}")


# --- Vectorization and Add to ChromaDB ---
def generate_chunk_id(text_chunk):  # ID based only on content
    """Generates a unique ID for a text chunk using SHA256 hashing."""
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()  # Added encoding


def embed_and_add_chunks_to_db(document_chunks_text, collection, retriever):
    """Embeds text chunks and adds them to ChromaDB, avoiding duplicates."""
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = len(document_chunks_text)
    print(f"Starting embedding process for {total_chunks} chunks...")
    for i, chunk_text in enumerate(document_chunks_text):
        chunk_id = generate_chunk_id(chunk_text)
        try:
            # Check if chunk ID already exists efficiently
            existing = collection.get(
                ids=[chunk_id], include=[]
            )  # Don't include embeddings/docs
            if existing["ids"]:
                # print(f"Chunk {i} (ID {chunk_id[:10]}...) exists. Skipping.")
                skipped_count += 1
            else:
                chunk_embedding = retriever.vectorize_text(chunk_text)
                collection.add(
                    embeddings=chunk_embedding, documents=[chunk_text], ids=[chunk_id]
                )
                # print(f"Chunk {i} (ID {chunk_id[:10]}...) embedded and added.")
                added_count += 1
            # Simple progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                 print(f"  Processed {i + 1}/{total_chunks} chunks (Added: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")

        except Exception as e:
            print(f"!!! ERROR processing chunk {i} (ID: {chunk_id[:10]}...): {e}")
            error_count += 1
            # Optionally continue or break depending on severity
            # continue
    print(
        f"Embedding finished. Added: {added_count}, Skipped (already exist): {skipped_count}, Errors: {error_count}"
    )


# --- Global Variables / Setup (Needed by importer and main block) ---

# Load config once
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
else:
    raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")
rag_params = config.get("rag_parameters", {})

# Initialize retriever instance (can be imported by rag_tester)
retriever = initialize_retriever(rag_params.get("retrieval_algorithm", "embedding"))
tokenizer = retriever.tokenizer if hasattr(retriever, 'tokenizer') else None

# Initialize embedding function for ChromaDB
gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="Alibaba-NLP/gte-Qwen2-7B-instruct")

# Initialize ChromaDB client and collection (can be imported by rag_tester)
chroma_client = chromadb.PersistentClient(path=persist_directory)
try:
    # Use get_or_create for simplicity
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=gte_embedding_function # Still associate EF for consistency
    )
    print(f"Collection '{collection_name}' loaded/created successfully.")
except Exception as e:
     print(f"!!! ERROR initializing ChromaDB collection: {e}")
     # Decide how to proceed - maybe exit or raise
     raise


# --- Main Execution Block ---
# This code only runs when rag_pipeline.py is executed directly
if __name__ == "__main__":

    print("\n--- Running RAG Pipeline Setup (Embedding) ---")

    # Ensure tokenizer is available if needed for splitting
    if not tokenizer and rag_params.get("retrieval_algorithm", "embedding") == "embedding":
         raise TypeError("Tokenizer not found in retriever, but required for token splitting.")

    # --- Document Loading and Chunking ---
    print(f"Loading text from: {path_to_text}")
    with open(path_to_text, 'r', encoding='utf-8') as f:
        text = f.read()

    # Define the token-based splitter
    token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=rag_params.get("chunk_size", 2000),
        chunk_overlap=rag_params.get("overlap_size", 50)
    )

    print(f"Splitting text using token limits: chunk_size={rag_params.get('chunk_size', 2000)}, overlap={rag_params.get('overlap_size', 50)}")
    document_chunks_text = token_splitter.split_text(text)
    print(f"Generated {len(document_chunks_text)} token-based chunks.")

    # --- Embedding and Adding Chunks to DB ---
    embed_and_add_chunks_to_db(document_chunks_text, collection, retriever)

    print("\n--- Running RAG Pipeline Evaluation (Retrieval Metrics) ---")
    # --- Load and Evaluate Datasets ---
    all_evaluation_results = {}
    for dataset_name, dataset_path in dataset_paths.items():
        dataset = load_dataset(dataset_path)
        if dataset:
            print(f"\n--- Evaluating Retrieval on {dataset_name} dataset ---")
            evaluation_results = evaluate_rag_pipeline(dataset, retriever, collection, rag_params)
            all_evaluation_results[dataset_name] = evaluation_results
            analyze_evaluation_results(evaluation_results, dataset_name)

    # --- Analyze Datasets ---
    analyze_dataset_across_types(dataset_paths)

    # --- Overall Analysis ---
    print("\n--- Overall Retrieval Evaluation Analysis ---")
    for dataset_name, results in all_evaluation_results.items():
        if "source_hit_rate" in results:
            print(f"Dataset: {dataset_name}, Source Hit Rate: {results['source_hit_rate']:.2f}%")
        else:
            print(f"Dataset: {dataset_name}, Source Hit Rate: Not Available")

    print("\n--- RAG Pipeline Script Finished ---")