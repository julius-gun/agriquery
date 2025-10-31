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
# Import for standalone execution
from utils.config_loader import ConfigLoader
from utils.chroma_embedding_function import HuggingFaceEmbeddingFunction

# --- Configuration Constants ---
persist_directory = "chroma_db"
config_file_path = "config_fast.json"  # Path to config file
MANUALS_DIRECTORY = "manuals"

# --- Helper Functions ---
def generate_chunk_id(text_chunk: str) -> str:
    """Generates a unique ID for a text chunk."""
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()

def initialize_retriever(
    retrieval_strategy_str: str,
    embedding_model_config: Dict[str, Any],
    chroma_client: Optional[chromadb.ClientAPI] = None,
    collection_name: Optional[str] = None,
    shared_embedding_retriever: Optional[EmbeddingRetriever] = None
) -> BaseRetriever:
    """
    Initializes the retriever based on the specified strategy, prioritizing
    the use of a shared embedding model instance.

    Args:
        retrieval_strategy_str (str): The retrieval strategy ('embedding', 'keyword', 'hybrid').
        embedding_model_config (Dict[str, Any]): Configuration for the embedding model (used as fallback).
        chroma_client (Optional[chromadb.ClientAPI]): The ChromaDB client instance.
        collection_name (Optional[str]): The name of the ChromaDB collection.
        shared_embedding_retriever (Optional[EmbeddingRetriever]): A pre-initialized
            EmbeddingRetriever instance.

    Returns:
        BaseRetriever: An instance of the specified retriever.
    """
    print(f"Initializing retriever for strategy: '{retrieval_strategy_str}'...")
    if retrieval_strategy_str == "embedding":
        if shared_embedding_retriever:
            print("  Using shared embedding retriever instance.")
            return shared_embedding_retriever
        else:
            print("  Creating new embedding retriever instance (fallback).")
            return EmbeddingRetriever(model_config=embedding_model_config)

    elif retrieval_strategy_str == "keyword":
        return KeywordRetriever()

    elif retrieval_strategy_str == "hybrid":
        if not chroma_client or not collection_name:
            raise ValueError("HybridRetriever requires chroma_client and collection_name during initialization.")
        
        if shared_embedding_retriever:
            print("  Initializing HybridRetriever with shared embedding retriever.")
            # This assumes HybridRetriever is updated to accept an instance
            return HybridRetriever(
                embedding_retriever=shared_embedding_retriever, # Pass the instance
                chroma_client=chroma_client,
                collection_name=collection_name
            )
        else:
            print("  Initializing HybridRetriever with new embedding retriever (fallback).")
            return HybridRetriever(
                embedding_model_config=embedding_model_config, # Fallback to config
                chroma_client=chroma_client,
                collection_name=collection_name
            )
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy_str}")


# This function's logic remains sound and does not need changes.
def embed_and_add_chunks_to_db(
    document_chunks_text: List[str],
    collection: chromadb.Collection,
    retriever: BaseRetriever # Use BaseRetriever type hint
) -> None:
    """Embeds text chunks and adds them to ChromaDB, avoiding duplicates WITHIN this specific collection."""
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = len(document_chunks_text)
    collection_name = collection.name # Get name for logging
    print(f"Starting embedding process for {total_chunks} chunks into collection '{collection_name}'...")

    is_embedding_retriever = isinstance(retriever, EmbeddingRetriever)
    internal_embedding_retriever = getattr(retriever, 'embedding_retriever', None) if isinstance(retriever, HybridRetriever) else None

    if not is_embedding_retriever and not internal_embedding_retriever:
        print(f"Warning: The provided retriever ({type(retriever).__name__}) may not be suitable for generating embeddings required by ChromaDB 'add'. Proceeding by adding documents only.")
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
        return

    embedder_to_use = retriever if is_embedding_retriever else internal_embedding_retriever

    for i, chunk_text in enumerate(document_chunks_text):
        chunk_id = generate_chunk_id(chunk_text)
        try:
            existing = collection.get(
                ids=[chunk_id], include=[]
            )
            if existing["ids"]:
                skipped_count += 1
            else:
                chunk_embedding = embedder_to_use.vectorize_document(chunk_text)
                if isinstance(chunk_embedding, list) and isinstance(chunk_embedding[0], list) and isinstance(chunk_embedding[0][0], float):
                     collection.add(
                         embeddings=chunk_embedding,
                         documents=[chunk_text],
                         ids=[chunk_id]
                     )
                     added_count += 1
                else:
                     print(f"Warning: Chunk {i} embedding format unexpected ({type(chunk_embedding)}). Adding document only.")
                     collection.add(
                         documents=[chunk_text],
                         ids=[chunk_id]
                     )
                     added_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                 print(f"  Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks (Added w/ Embeddings: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")
        except Exception as e:
            print(f"!!! ERROR processing chunk {i} for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
            error_count += 1
    print(
        f"Embedding finished for collection '{collection_name}'. Added: {added_count}, Skipped (already exist): {skipped_count}, Errors: {error_count}"
    )


# --- Main Execution Block ---
# This code only runs when rag_pipeline.py is executed directly
if __name__ == "__main__":
    
    print("\n--- Running RAG Pipeline Script in Standalone Mode ---")
    
    # --- Step 1: Initialize all components needed for standalone execution ---
    print("\n--- Initializing Standalone Components ---")
    try:
        # Load config
        config_loader = ConfigLoader(config_file_path)
        config = config_loader.config
        
        # Load specific config sections
        embedding_model_config = config_loader.get_embedding_model_config()
        rag_params = config_loader.get_rag_parameters()
        files_to_test = config_loader.get_files_to_test()
        file_extensions_to_test = config_loader.get_file_extensions_to_test()
        dataset_paths = config_loader.get_question_dataset_paths()

        # Create the single EmbeddingRetriever instance for this script run
        print("Initializing standalone EmbeddingRetriever...")
        standalone_retriever = EmbeddingRetriever(model_config=embedding_model_config)
        tokenizer: Any = standalone_retriever.tokenizer if hasattr(standalone_retriever, 'tokenizer') else None
        
        # Create the custom ChromaDB embedding function using the retriever
        print("Initializing custom Chroma embedding function...")
        standalone_embedding_function = HuggingFaceEmbeddingFunction(embedding_retriever=standalone_retriever)
        
        # Initialize ChromaDB client
        print("Initializing ChromaDB client...")
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        print("--- Standalone Initialization Complete ---\n")
    
    except Exception as e:
        print(f"FATAL: Failed to initialize components for standalone run. Error: {e}")
        exit(1)


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
                            embedding_function=standalone_embedding_function
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
                            embedding_function=standalone_embedding_function
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
                        
                        # Use the single retriever instance created at the start
                        embed_and_add_chunks_to_db(document_chunks_text, collection, standalone_retriever)
                    else:
                        print(f"Skipping processing for existing collection '{dynamic_collection_name}'.")

                    # --- Evaluation (Optional, run against the current collection) ---
                    print(f"\n--- Running RAG Pipeline Evaluation (Retrieval Metrics on '{dynamic_collection_name}') ---")

                    if collection is None:
                         try:
                             collection = chroma_client.get_collection(
                                 name=dynamic_collection_name, 
                                 embedding_function=standalone_embedding_function
                             )
                         except Exception:
                              print(f"Error: Collection object '{dynamic_collection_name}' not available for evaluation. Skipping.")
                              continue

                    all_evaluation_results_for_combo = {}
                    for dataset_name, dataset_path in dataset_paths.items():
                        dataset = load_dataset(dataset_path)
                        if dataset:
                            print(f"\n--- Evaluating Retrieval on {dataset_name} dataset using collection '{dynamic_collection_name}' ---")
                            # Use the single retriever instance for evaluation as well
                            evaluation_results = evaluate_rag_pipeline(dataset, standalone_retriever, collection, rag_params)
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
