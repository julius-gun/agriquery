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
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
persist_directory = "chroma_db"
config_file_path = "config.json"  # Path to config file

# --- Helper Functions (generate_chunk_id, initialize_retriever, embed_and_add_chunks_to_db remain the same) ---
def generate_chunk_id(text_chunk):
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()

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

def embed_and_add_chunks_to_db(document_chunks_text, collection, retriever):
    """Embeds text chunks and adds them to ChromaDB, avoiding duplicates WITHIN this specific collection."""
    # This function remains largely the same, but operates on the specific collection passed to it.
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = len(document_chunks_text)
    collection_name = collection.name # Get name for logging
    print(f"Starting embedding process for {total_chunks} chunks into collection '{collection_name}'...")
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
                 print(f"  Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks (Added: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")
        except Exception as e:
            print(f"!!! ERROR processing chunk {i} for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
            error_count += 1
            # Optionally continue or break depending on severity
            # continue
    print(
        f"Embedding finished for collection '{collection_name}'. Added: {added_count}, Skipped (already exist): {skipped_count}, Errors: {error_count}"
    )


# --- Global Variables / Setup (Client, Embedding Function, Config) ---
# Load config once
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
else:
    raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")

# --- Load specific config sections ---
rag_params = config.get("rag_parameters", {})
language_configs = config.get("language_configs", []) # Load language configs
dataset_paths = config.get("question_dataset_paths", {}) # Load English question paths

if not language_configs:
    raise ValueError("No 'language_configs' found in config.json. Please define language-specific settings.")
if not dataset_paths:
    raise ValueError("No 'question_dataset_paths' found in config.json.")


# Initialize retriever instance (can be imported by rag_tester)
# Retriever choice doesn't directly affect collection name based on chunking
retriever = initialize_retriever(rag_params.get("retrieval_algorithm", "embedding"))
tokenizer = retriever.tokenizer if hasattr(retriever, 'tokenizer') else None

# Initialize embedding function for ChromaDB
gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="Alibaba-NLP/gte-Qwen2-7B-instruct")

# Initialize ChromaDB client (can be imported by rag_tester)
chroma_client = chromadb.PersistentClient(path=persist_directory)

# --- Main Execution Block ---
# This code only runs when rag_pipeline.py is executed directly
if __name__ == "__main__":

    print("\n--- Running RAG Pipeline Setup (Embedding & Evaluation) for Configured Languages ---")

    # --- Get common RAG parameters ---
    chunk_size = rag_params.get("chunk_size", 2000) # Get current chunk size
    overlap_size = rag_params.get("overlap_size", 50) # Get current overlap size

    # --- Loop through each language configuration ---
    for lang_config in language_configs:
        language = lang_config.get("language")
        manual_path = lang_config.get("manual_path")
        base_collection_name = lang_config.get("collection_base_name")

        if not all([language, manual_path, base_collection_name]):
            print(f"Warning: Skipping invalid language config entry: {lang_config}")
            continue

        print(f"\n===== Processing Language: {language.upper()} =====")
        print(f"Manual Path: {manual_path}")

        # --- Determine Dynamic Collection Name for this language ---
        dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
        print(f"Target collection name: '{dynamic_collection_name}'")

        # --- Check if Collection Exists ---
        collection_exists = False
        collection = None # Initialize collection variable
        try:
            # Try to get the collection. If it succeeds, it exists.
            collection = chroma_client.get_collection(
                name=dynamic_collection_name,
                embedding_function=gte_embedding_function # Provide EF for potential validation
            )
            print(f"Collection '{dynamic_collection_name}' already exists.")
            collection_exists = True
        except Exception as e:
            # Assuming exception means it doesn't exist (adjust based on specific ChromaDB exceptions if needed)
            print(f"Collection '{dynamic_collection_name}' does not exist yet. Will proceed with creation and embedding.")
            collection_exists = False

        # --- Conditional Embedding ---
        if not collection_exists:
            print(f"Creating new collection: '{dynamic_collection_name}'")
            # Create the collection explicitly
            collection = chroma_client.create_collection(
                name=dynamic_collection_name,
                embedding_function=gte_embedding_function
            )

            # Ensure tokenizer is available if needed for splitting
            if not tokenizer and rag_params.get("retrieval_algorithm", "embedding") == "embedding":
                 raise TypeError("Tokenizer not found in retriever, but required for token splitting.")

            # --- Document Loading and Chunking for this language ---
            print(f"Loading text from: {manual_path}")
            try:
                with open(manual_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except FileNotFoundError:
                print(f"!!! ERROR: Manual file not found at '{manual_path}'. Skipping embedding for {language}.")
                continue # Skip to the next language

            # Define the token-based splitter using current parameters
            token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=chunk_size, # Use loaded chunk_size
                chunk_overlap=overlap_size # Use loaded overlap_size
            )

            print(f"Splitting text using token limits: chunk_size={chunk_size}, overlap={overlap_size}")
            document_chunks_text = token_splitter.split_text(text)
            print(f"Generated {len(document_chunks_text)} token-based chunks for {language}.")

            # --- Embedding and Adding Chunks to the NEW DB ---
            embed_and_add_chunks_to_db(document_chunks_text, collection, retriever)
        else:
            print(f"Skipping embedding process for existing collection '{dynamic_collection_name}'.")

        # --- Evaluation (Optional, run against the current language's collection) ---
        # Note: Evaluation in rag_pipeline.py usually focuses on retrieval metrics (like source hit rate)
        # which depend on the specific collection content.
        print(f"\n--- Running RAG Pipeline Evaluation (Retrieval Metrics on '{dynamic_collection_name}') ---")

        # Ensure 'collection' variable refers to the correct one (either newly created or retrieved)
        if collection is None:
             print(f"Error: Collection object '{dynamic_collection_name}' not available for evaluation. Skipping evaluation for {language}.")
             continue # Skip evaluation for this language

        # Use the English question datasets for evaluation against the current language collection
        all_evaluation_results_for_lang = {}
        for dataset_name, dataset_path in dataset_paths.items():
            dataset = load_dataset(dataset_path) # Load English questions
            if dataset:
                print(f"\n--- Evaluating Retrieval on {dataset_name} dataset (English Qs) using collection '{dynamic_collection_name}' ---")
                # Pass the specific collection object for the current language
                evaluation_results = evaluate_rag_pipeline(dataset, retriever, collection, rag_params)
                all_evaluation_results_for_lang[dataset_name] = evaluation_results
                analyze_evaluation_results(evaluation_results, f"{dataset_name} ({language} manual)") # Add language context to analysis output

        # --- Analyze Datasets (English questions) ---
        # This analysis is about the question sets themselves, doesn't depend on the language collection
        # analyze_dataset_across_types(dataset_paths) # Might be redundant to call this in every loop iteration

        # --- Overall Analysis for this Language Collection ---
        print(f"\n--- Overall Retrieval Evaluation Analysis for Collection '{dynamic_collection_name}' ({language.upper()}) ---")
        for dataset_name, results in all_evaluation_results_for_lang.items():
            if "source_hit_rate" in results:
                print(f"  Dataset: {dataset_name}, Source Hit Rate: {results['source_hit_rate']:.2f}%")
            else:
                print(f"  Dataset: {dataset_name}, Source Hit Rate: Not Available")

    # Analyze English datasets once after the loop
    print("\n--- Analysis of English Question Datasets Used ---")
    analyze_dataset_across_types(dataset_paths)

    print("\n--- RAG Pipeline Script Finished ---")