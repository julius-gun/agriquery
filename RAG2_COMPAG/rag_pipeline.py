import chromadb
from chromadb.utils import embedding_functions
import hashlib
# Import specific retrievers
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
from retrieval_pipelines.hybrid_retriever import HybridRetriever # Import HybridRetriever
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
from typing import List, Dict, Any, Optional # Import Optional

# --- Configuration ---
persist_directory = "chroma_db"
config_file_path = "config.json"  # Path to config file
MANUALS_DIRECTORY = "manuals"

# --- Helper Functions ---
def generate_chunk_id(text_chunk: str) -> str:
    """Generates a unique ID for a text chunk."""
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()

# Update return type hint to BaseRetriever
# Update signature to accept chroma_client and collection_name
def initialize_retriever(
    retrieval_strategy_str: str,
    chroma_client: Optional[chromadb.ClientAPI] = None, # Add chroma_client
    collection_name: Optional[str] = None # Add collection_name
) -> BaseRetriever:
    """
    Initializes the retriever based on the specified strategy.

    Args:
        retrieval_strategy_str (str): The retrieval strategy ('embedding', 'keyword', 'hybrid').
        chroma_client (Optional[chromadb.ClientAPI]): The ChromaDB client instance (required for hybrid).
        collection_name (Optional[str]): The name of the ChromaDB collection (required for hybrid).

    Returns:
        BaseRetriever: An instance of the specified retriever.

    Raises:
        ValueError: If the strategy is unknown or required arguments are missing for hybrid.
        NotImplementedError: If the strategy is 'hybrid' (placeholder).
    """
    print(f"Initializing retriever for strategy: '{retrieval_strategy_str}'...") # Added logging
    if retrieval_strategy_str == "embedding":
        # EmbeddingRetriever doesn't strictly need client/collection at init
        # It's used externally in rag_tester for querying ChromaDB
        return EmbeddingRetriever() # Assuming default params are okay here
    elif retrieval_strategy_str == "keyword":
        # KeywordRetriever doesn't need client/collection at init
        return KeywordRetriever()
    elif retrieval_strategy_str == "hybrid":
        # Hybrid retriever *does* need client and collection name for its embedding part
        if not chroma_client or not collection_name:
             raise ValueError("HybridRetriever requires chroma_client and collection_name during initialization.")
        # Ensure HybridRetriever is imported (already done above)
        # from retrieval_pipelines.hybrid_retriever import HybridRetriever
        # TODO: Make embedding model/max_length configurable if needed
        return HybridRetriever(chroma_client=chroma_client, collection_name=collection_name)
    # elif retrieval_strategy_str == "some_other_future_strategy":
    #     raise NotImplementedError("This other strategy is not yet implemented.")
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy_str}")

# Update retriever parameter type hint to BaseRetriever
def embed_and_add_chunks_to_db(
    document_chunks_text: List[str],
    collection: chromadb.Collection,
    retriever: BaseRetriever # Use BaseRetriever type hint
) -> None:
    # Note: This function is primarily for *populating* the DB.
    # HybridRetriever.vectorize_text returns a dict, but ChromaDB `add` expects embeddings.
    # This function might need adjustment if used directly with a HybridRetriever instance
    # for populating, or we assume it's only called with EmbeddingRetriever during DB creation.
    # Let's assume `create_databases.py` or the main block here uses EmbeddingRetriever for population.
    """Embeds text chunks and adds them to ChromaDB, avoiding duplicates WITHIN this specific collection."""
    # This function remains largely the same, but operates on the specific collection passed to it.
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = len(document_chunks_text)
    collection_name = collection.name # Get name for logging
    print(f"Starting embedding process for {total_chunks} chunks into collection '{collection_name}'...")

    # Determine if the retriever is embedding-capable for adding to Chroma
    is_embedding_retriever = isinstance(retriever, EmbeddingRetriever)
    # HybridRetriever *contains* an EmbeddingRetriever, so we might need to access it
    internal_embedding_retriever = getattr(retriever, 'embedding_retriever', None) if isinstance(retriever, HybridRetriever) else None

    if not is_embedding_retriever and not internal_embedding_retriever:
        print(f"Warning: The provided retriever ({type(retriever).__name__}) may not be suitable for generating embeddings required by ChromaDB 'add'. Proceeding by adding documents only.")
        # Fallback: Add documents only if no embedding capability found
        for i, chunk_text in enumerate(document_chunks_text):
            chunk_id = generate_chunk_id(chunk_text)
            try:
                existing = collection.get(ids=[chunk_id], include=[])
                if existing["ids"]:
                    skipped_count += 1
                else:
                    collection.add(documents=[chunk_text], ids=[chunk_id])
                    added_count += 1
                if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                    print(f"  Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks (Added Docs Only: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")
            except Exception as e:
                print(f"!!! ERROR processing chunk {i} (doc only) for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
                error_count += 1
        print(f"Document-only adding finished for collection '{collection_name}'. Added: {added_count}, Skipped: {skipped_count}, Errors: {error_count}")
        return # Exit function after document-only add

    # Proceed with embedding if retriever is suitable
    embedder_to_use = retriever if is_embedding_retriever else internal_embedding_retriever

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
                # Use the appropriate embedder's vectorize_text method
                # Assuming it returns List[List[float]]
                chunk_embedding = embedder_to_use.vectorize_text(chunk_text)

                if isinstance(chunk_embedding, list) and isinstance(chunk_embedding[0], list) and isinstance(chunk_embedding[0][0], float):
                     # Standard embedding format
                     collection.add(
                         embeddings=chunk_embedding, # Should be List[List[float]]
                         documents=[chunk_text],
                         ids=[chunk_id]
                     )
                     added_count += 1
                else:
                     # Handle unexpected format from embedder
                     print(f"Warning: Chunk {i} embedding format unexpected ({type(chunk_embedding)}). Adding document only.")
                     collection.add(
                         documents=[chunk_text],
                         ids=[chunk_id]
                         # Potentially add metadata if vectorize_text produced keywords, etc.
                         # metadatas=[{"keywords": chunk_representation}] # Example
                     )
                     added_count += 1 # Or handle differently

            if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                 print(f"  Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks (Added w/ Embeddings: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")
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
files_to_test = config.get("files_to_test", [])
file_extensions_to_test = config.get("file_extensions_to_test", [])
dataset_paths = config.get("question_dataset_paths", {})

if not files_to_test:
    raise ValueError("No 'files_to_test' found in config.json.")
if not file_extensions_to_test:
    raise ValueError("No 'file_extensions_to_test' found in config.json.")
if not dataset_paths:
    raise ValueError("No 'question_dataset_paths' found in config.json.")


# Initialize retriever instance (can be imported by rag_tester)
temp_embed_retriever = EmbeddingRetriever()
tokenizer: Any = temp_embed_retriever.tokenizer if hasattr(temp_embed_retriever, 'tokenizer') else None


# Initialize embedding function for ChromaDB (specific to embedding models)
embedding_model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct" # TODO: Potentially make configurable if needed elsewhere
gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

# Initialize ChromaDB client (can be imported by rag_tester)
chroma_client = chromadb.PersistentClient(path=persist_directory)

# --- Main Execution Block ---
# This code only runs when rag_pipeline.py is executed directly
if __name__ == "__main__":

    print("\n--- Running RAG Pipeline Setup (Embedding & Evaluation) for Configured Files ---")

    chunk_sizes_to_process = rag_params.get("chunk_sizes_to_test", [2000])
    overlap_sizes_to_process = rag_params.get("overlap_sizes_to_test", [50])

    for chunk_size in chunk_sizes_to_process:
        for overlap_size in overlap_sizes_to_process:
            print(f"\n{'='*10} Processing for Chunk Size: {chunk_size}, Overlap: {overlap_size} {'='*10}")

            for file_basename in files_to_test:
                for extension in file_extensions_to_test:
                    manual_path = os.path.join(MANUALS_DIRECTORY, f"{file_basename}.{extension}")
                    if not os.path.isfile(manual_path):
                        continue

                    print(f"\n===== Processing File: {os.path.basename(manual_path)} =====")

                    sanitized_ext = extension.replace('.', '_')
                    base_collection_name = f"{file_basename}_{sanitized_ext}"
                    dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
                    print(f"Target collection name: '{dynamic_collection_name}'")

                    collection_exists = False
                    collection = None
                    try:
                        collection = chroma_client.get_collection(
                            name=dynamic_collection_name,
                            embedding_function=gte_embedding_function
                        )
                        print(f"Collection '{dynamic_collection_name}' already exists.")
                        collection_exists = True
                    except Exception as e:
                        print(f"Collection '{dynamic_collection_name}' does not exist yet. Will proceed with creation.")
                        collection_exists = False

                    if not collection_exists:
                        print(f"Creating new collection: '{dynamic_collection_name}'")
                        collection = chroma_client.create_collection(
                            name=dynamic_collection_name,
                            embedding_function=gte_embedding_function
                        )

                        if not tokenizer:
                             print("Warning: Tokenizer not found, required for token splitting.")

                        print(f"Loading text from: {manual_path}")
                        try:
                            with open(manual_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                        except FileNotFoundError:
                            print(f"!!! ERROR: Manual file not found at '{manual_path}'. Skipping.")
                            continue

                        if tokenizer:
                            token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                                tokenizer=tokenizer,
                                chunk_size=chunk_size,
                                chunk_overlap=overlap_size
                            )
                            print(f"Splitting text using token limits: chunk_size={chunk_size}, overlap={overlap_size}")
                            document_chunks_text = token_splitter.split_text(text)
                            print(f"Generated {len(document_chunks_text)} token-based chunks.")
                        else:
                            print(f"Warning: Tokenizer not available. Using basic character splitter.")
                            char_splitter = RecursiveCharacterTextSplitter(
                                 chunk_size=chunk_size * 4,
                                 chunk_overlap=overlap_size * 4,
                                 length_function=len,
                                 is_separator_regex=False,
                            )
                            document_chunks_text = char_splitter.split_text(text)
                            print(f"Generated {len(document_chunks_text)} character-based chunks.")

                        db_populating_retriever = EmbeddingRetriever()
                        embed_and_add_chunks_to_db(document_chunks_text, collection, db_populating_retriever)
                    else:
                        print(f"Skipping processing for existing collection '{dynamic_collection_name}'.")

                    # --- Evaluation (Optional, run against the current collection) ---
                    print(f"\n--- Running RAG Pipeline Evaluation (Retrieval Metrics on '{dynamic_collection_name}') ---")

                    if collection is None:
                         try:
                             collection = chroma_client.get_collection(name=dynamic_collection_name, embedding_function=gte_embedding_function)
                         except Exception:
                              print(f"Error: Collection object '{dynamic_collection_name}' not available for evaluation. Skipping.")
                              continue

                    all_evaluation_results_for_combo = {}
                    for dataset_name, dataset_path in dataset_paths.items():
                        dataset = load_dataset(dataset_path)
                        if dataset:
                            print(f"\n--- Evaluating Retrieval on {dataset_name} dataset using collection '{dynamic_collection_name}' ---")
                            eval_retriever = EmbeddingRetriever()
                            evaluation_results = evaluate_rag_pipeline(dataset, eval_retriever, collection, rag_params)
                            all_evaluation_results_for_combo[dataset_name] = evaluation_results
                            analysis_label = f"{dataset_name} ({os.path.basename(manual_path)}, CS={chunk_size}, OS={overlap_size})"
                            analyze_evaluation_results(evaluation_results, analysis_label)

                    print(f"\n--- Overall Retrieval Evaluation Analysis for Collection '{dynamic_collection_name}' ---")
                    for dataset_name, results in all_evaluation_results_for_combo.items():
                         hit_rate = results.get('source_hit_rate', 'N/A')
                         if isinstance(hit_rate, float):
                             print(f"  Dataset: {dataset_name}, Source Hit Rate: {hit_rate:.2f}%")
                         else:
                             print(f"  Dataset: {dataset_name}, Source Hit Rate: {hit_rate}")


    # Analyze English datasets once after all loops
    print("\n--- Analysis of English Question Datasets Used ---")
    analyze_dataset_across_types(dataset_paths)

    print("\n--- RAG Pipeline Script Finished ---")
