import chromadb
from chromadb.utils import embedding_functions
import hashlib
# Removed KeywordRetriever import here, as it's only used inside initialize_retriever
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
# Import BaseRetriever for type hinting
from retrieval_pipelines.base_retriever import BaseRetriever
from evaluation.evaluation_metrics import evaluate_rag_pipeline
from analysis.analysis_tools import (
    analyze_evaluation_results,
    load_dataset,
    analyze_dataset_across_types,
)
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any # Import Any for type hinting

# --- Configuration ---
persist_directory = "chroma_db"
config_file_path = "config.json"  # Path to config file

# --- Helper Functions ---
def generate_chunk_id(text_chunk: str) -> str:
    """Generates a unique ID for a text chunk."""
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()

# Update return type hint to BaseRetriever
def initialize_retriever(retrieval_strategy_str: str) -> BaseRetriever:
    """Initializes the retriever based on the specified strategy."""
    if retrieval_strategy_str == "embedding":
        # Ensure EmbeddingRetriever is imported (already done above or can be moved here)
        # from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
        return EmbeddingRetriever() # Assuming default params are okay here
    elif retrieval_strategy_str == "keyword":
        # Ensure KeywordRetriever is imported
        from retrieval_pipelines.keyword_retrieval import KeywordRetriever
        return KeywordRetriever()
    elif retrieval_strategy_str == "hybrid":
        raise NotImplementedError("Hybrid retrieval not yet implemented.")
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy_str}")

# Update retriever parameter type hint to BaseRetriever
def embed_and_add_chunks_to_db(
    document_chunks_text: List[str],
    collection: chromadb.Collection,
    retriever: BaseRetriever # Use BaseRetriever type hint
) -> None:
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
                # Polymorphic call to vectorize_text
                chunk_representation = retriever.vectorize_text(chunk_text)
                # Assuming embedding retriever returns List[List[float]] and we take the first/only element
                # And assuming keyword retriever returns something compatible or this logic needs adjustment
                # For ChromaDB add, we need 'embeddings' (list of lists) or 'documents'
                # This part might need refinement depending on what KeywordRetriever.vectorize_text returns
                # and how non-embedding data should be stored.
                # Current ChromaDB `add` expects embeddings OR documents.
                # Let's assume for now this script primarily handles embedding storage.
                # If KeywordRetriever is used, this `add` call might fail if chunk_representation isn't embedding-like.
                # A more robust approach might be needed if keyword data needs storing differently.
                if isinstance(chunk_representation, list) and isinstance(chunk_representation[0], list) and isinstance(chunk_representation[0][0], float):
                     # Likely an embedding
                     collection.add(
                         embeddings=chunk_representation, # Should be List[List[float]]
                         documents=[chunk_text],
                         ids=[chunk_id]
                     )
                     added_count += 1
                else:
                     # Handle non-embedding case or log a warning/error
                     # Maybe just add the document?
                     print(f"Warning: Chunk {i} representation is not a standard embedding. Adding document only.")
                     collection.add(
                         documents=[chunk_text],
                         ids=[chunk_id]
                         # Potentially add metadata if vectorize_text produced keywords, etc.
                         # metadatas=[{"keywords": chunk_representation}] # Example
                     )
                     added_count += 1 # Or handle differently

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
retriever: BaseRetriever = initialize_retriever(rag_params.get("retrieval_algorithm", "embedding"))
# Use Any for tokenizer if we stick with hasattr, or define getter in BaseRetriever
tokenizer: Any = retriever.tokenizer if hasattr(retriever, 'tokenizer') else None

# Initialize embedding function for ChromaDB (specific to embedding models)
# This might need conditional logic if non-embedding models are primary
embedding_model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct" # TODO: Potentially make configurable if needed elsewhere
gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

# Initialize ChromaDB client (can be imported by rag_tester)
chroma_client = chromadb.PersistentClient(path=persist_directory)

# --- Main Execution Block ---
# This code only runs when rag_pipeline.py is executed directly
if __name__ == "__main__":

    print("\n--- Running RAG Pipeline Setup (Embedding & Evaluation) for Configured Languages ---")

    # --- Get common RAG parameters ---
    # Allow overriding chunk/overlap from config if needed, otherwise use defaults
    # These parameters are now primarily used for naming collections and splitting text.
    chunk_sizes_to_process = rag_params.get("chunk_sizes_to_test", [2000]) # Default if not in config
    overlap_sizes_to_process = rag_params.get("overlap_sizes_to_test", [50]) # Default if not in config

    # --- Loop through chunk/overlap combinations defined in config ---
    # This script now creates DBs for all combinations specified in the config's test parameters
    for chunk_size in chunk_sizes_to_process:
        for overlap_size in overlap_sizes_to_process:
            print(f"\n{'='*10} Processing for Chunk Size: {chunk_size}, Overlap: {overlap_size} {'='*10}")

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

                # --- Determine Dynamic Collection Name ---
                dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
                print(f"Target collection name: '{dynamic_collection_name}'")

        # --- Check if Collection Exists ---
                collection_exists = False
                collection = None
                try:
                    # Use the specific embedding function when getting/creating embedding collections
                    collection = chroma_client.get_collection(
                        name=dynamic_collection_name,
                        embedding_function=gte_embedding_function # Needed if collection stores embeddings
                    )
                    print(f"Collection '{dynamic_collection_name}' already exists.")
                    collection_exists = True
                except Exception as e:
                    print(f"Collection '{dynamic_collection_name}' does not exist yet. Will proceed with creation.")
                    collection_exists = False

                # --- Conditional Embedding/Processing ---
                if not collection_exists:
                    print(f"Creating new collection: '{dynamic_collection_name}'")
            # Create the collection explicitly
                    collection = chroma_client.create_collection(
                        name=dynamic_collection_name,
                        embedding_function=gte_embedding_function # Specify EF at creation
                        # Add metadata if needed: metadata={"hnsw:space": "cosine"} # Example
                    )

                    # Ensure tokenizer is available if needed for splitting
                    # This check is more relevant if using token splitting strategy
                    if not tokenizer and isinstance(retriever, EmbeddingRetriever): # Check specific type if tokenizer is specific
                         # Or check based on a method/property defined in BaseRetriever?
                         print("Warning: Tokenizer not found in retriever, required for token splitting.")
                         # Decide if this is fatal or if an alternative splitter can be used
                         # For now, let's assume RecursiveCharacterTextSplitter needs it.
                         # If rag_tester uses a different retriever, this might be okay.
                         # Let's try to proceed but log the warning.
                         # raise TypeError("Tokenizer not found in retriever, but required for token splitting.")


                    print(f"Loading text from: {manual_path}")
                    try:
                        with open(manual_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except FileNotFoundError:
                        print(f"!!! ERROR: Manual file not found at '{manual_path}'. Skipping processing for {language} / CS={chunk_size} / OS={overlap_size}.")
                        continue # Skip to the next language

                    # Define the splitter using current parameters
                    # Ensure tokenizer is available for this specific splitter
                    if tokenizer:
                        token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                            tokenizer=tokenizer,
                            chunk_size=chunk_size,
                            chunk_overlap=overlap_size
                        )
                        print(f"Splitting text using token limits: chunk_size={chunk_size}, overlap={overlap_size}")
                        document_chunks_text = token_splitter.split_text(text)
                        print(f"Generated {len(document_chunks_text)} token-based chunks for {language}.")
                    else:
                        # Fallback or error if tokenizer needed but not found
                        print(f"Warning: Tokenizer not available. Using basic character splitter.")
                        # Example fallback:
                        char_splitter = RecursiveCharacterTextSplitter(
                             chunk_size=chunk_size * 4, # Heuristic: multiply by avg token length
                             chunk_overlap=overlap_size * 4 # Heuristic
                        )
                        document_chunks_text = char_splitter.split_text(text)
                        print(f"Generated {len(document_chunks_text)} character-based chunks for {language}.")


                    # --- Embedding/Processing and Adding Chunks ---
                    # Pass the globally initialized retriever (could be embedding or keyword)
                    # The embed_and_add function now has logic to handle representation type
                    embed_and_add_chunks_to_db(document_chunks_text, collection, retriever)
                else:
                    print(f"Skipping processing for existing collection '{dynamic_collection_name}'.")

                # --- Evaluation (Optional, run against the current collection) ---
                # This evaluation likely assumes an embedding retriever/collection setup
                print(f"\n--- Running RAG Pipeline Evaluation (Retrieval Metrics on '{dynamic_collection_name}') ---")

                if collection is None:
                     # Try to get the collection again if it existed but wasn't assigned
                     try:
                         collection = chroma_client.get_collection(name=dynamic_collection_name, embedding_function=gte_embedding_function)
                     except Exception:
                          print(f"Error: Collection object '{dynamic_collection_name}' not available for evaluation. Skipping evaluation.")
                          continue # Skip evaluation

                # Use the English question datasets for evaluation
                all_evaluation_results_for_lang_combo = {}
                for dataset_name, dataset_path in dataset_paths.items():
                    dataset = load_dataset(dataset_path)
                    if dataset:
                        print(f"\n--- Evaluating Retrieval on {dataset_name} dataset using collection '{dynamic_collection_name}' ---")
                        # Pass the specific collection and the global retriever
                        # evaluate_rag_pipeline needs to handle the retriever type appropriately
                        evaluation_results = evaluate_rag_pipeline(dataset, retriever, collection, rag_params)
                        all_evaluation_results_for_lang_combo[dataset_name] = evaluation_results
                        analysis_label = f"{dataset_name} ({language} manual, CS={chunk_size}, OS={overlap_size})"
                        analyze_evaluation_results(evaluation_results, analysis_label)

                print(f"\n--- Overall Retrieval Evaluation Analysis for Collection '{dynamic_collection_name}' ---")
                for dataset_name, results in all_evaluation_results_for_lang_combo.items():
                     # Check if 'source_hit_rate' exists before accessing
                     hit_rate = results.get('source_hit_rate', 'N/A')
                     if isinstance(hit_rate, float):
                         print(f"  Dataset: {dataset_name}, Source Hit Rate: {hit_rate:.2f}%")
                     else:
                         print(f"  Dataset: {dataset_name}, Source Hit Rate: {hit_rate}")


    # Analyze English datasets once after all loops
    print("\n--- Analysis of English Question Datasets Used ---")
    analyze_dataset_across_types(dataset_paths)

    print("\n--- RAG Pipeline Script Finished ---")