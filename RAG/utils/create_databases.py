# create_databases.py
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import os
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys # Added for path adjustment
import pathlib # Added for robust path handling

# --- Adjust Python Path ---
# Add the project root directory (p_llm_manual/RAG) to the Python path
# This allows finding modules like retrieval_pipelines and utils
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
# --- End Path Adjustment ---


# Assuming these modules are accessible from the project root
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from utils.config_loader import ConfigLoader

# --- Configuration ---
PERSIST_DIRECTORY = "chroma_db"
# Config file path will be passed as an argument to main()

# --- Parameters for Database Creation ---
# These will now be loaded from the config file's "rag_parameters" section
# CHUNK_SIZES_TO_CREATE = [200, 500, 1000, 2000] # REMOVED - Load from config
# OVERLAP_SIZES_TO_CREATE = [100, 200, 500] # REMOVED - Load from config

# --- Helper Functions ---

def generate_chunk_id(text_chunk: str) -> str:
    """Generates a unique SHA256 hash ID for a text chunk."""
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()

def embed_and_add_chunks_to_db(document_chunks_text, collection, retriever):
    """
    Embeds text chunks using the provided retriever and adds them to the specified ChromaDB collection.
    Checks for existing chunk IDs within the collection before adding.
    """
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = len(document_chunks_text)
    collection_name = collection.name # Get name for logging
    print(f"  Starting embedding process for {total_chunks} chunks into collection '{collection_name}'...")
    start_time = time.time()

    ids_to_add = []
    embeddings_to_add = []
    documents_to_add = []

    # First, identify which chunks need embedding (check existence)
    try:
        existing_ids = set(collection.get(include=[])['ids']) # Get all existing IDs efficiently
    except Exception as e:
        print(f"  !!! WARNING: Could not retrieve existing IDs from collection '{collection_name}'. May attempt to re-add existing chunks. Error: {e}")
        existing_ids = set() # Proceed assuming no IDs exist, duplicates might be added if collection isn't empty

    for i, chunk_text in enumerate(document_chunks_text):
        chunk_id = generate_chunk_id(chunk_text)
        if chunk_id in existing_ids:
            skipped_count += 1
        else:
            try:
                # Vectorize only if not skipped
                chunk_embedding = retriever.vectorize_text(chunk_text)
                # Ensure chunk_embedding is a flat list of floats, not a list of lists
                if isinstance(chunk_embedding, list) and len(chunk_embedding) == 1 and isinstance(chunk_embedding[0], list):
                    embedding_vector = chunk_embedding[0]
                elif isinstance(chunk_embedding, list) and all(isinstance(item, float) for item in chunk_embedding):
                     embedding_vector = chunk_embedding # Already a flat list
                else:
                    # Handle unexpected format
                    print(f"  !!! WARNING: Unexpected embedding format for chunk {i}. Type: {type(chunk_embedding)}. Skipping chunk.")
                    error_count += 1
                    continue # Skip this chunk

                ids_to_add.append(chunk_id)
                embeddings_to_add.append(embedding_vector) # Add the actual embedding vector
                documents_to_add.append(chunk_text)
                # Don't increment added_count here, do it after successful batch add
                existing_ids.add(chunk_id) # Add to set to prevent duplicate adds in this run
            except Exception as e:
                print(f"  !!! ERROR vectorizing chunk {i} for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
                error_count += 1

        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == total_chunks:
             current_time = time.time()
             elapsed = current_time - start_time
             print(f"    Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks... "
                   f"(Checked: {skipped_count+len(ids_to_add)+error_count}, To Add: {len(ids_to_add)}, Errors: {error_count}) "
                   f"Elapsed: {elapsed:.2f}s")

    # Add collected chunks in batches
    if ids_to_add:
        num_added_in_batch = 0
        try:
            print(f"  Adding {len(ids_to_add)} new chunks to collection '{collection_name}'...")
            batch_size = 1000 # Adjust as needed
            for i in range(0, len(ids_to_add), batch_size):
                batch_ids = ids_to_add[i:i+batch_size]
                batch_embeddings = embeddings_to_add[i:i+batch_size]
                batch_documents = documents_to_add[i:i+batch_size]
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents
                )
                num_added_in_batch += len(batch_ids)
                print(f"    Added batch {i//batch_size + 1}/{(len(ids_to_add) + batch_size - 1)//batch_size} ({len(batch_ids)} chunks)")
            added_count = num_added_in_batch # Update added_count only on success
            print(f"  Successfully added {added_count} chunks.")
        except Exception as e:
            print(f"  !!! ERROR adding batch to collection '{collection_name}': {e}")
            # Note: This might leave the collection partially updated
            error_count += len(ids_to_add) # Count all potential adds as errors if batch fails
            # added_count remains 0 or its value before this batch attempt
            print(f"  !!! Batch add failed. Added count for this run remains {added_count}.")


    end_time = time.time()
    total_duration = end_time - start_time
    print(
        f"  Embedding finished for collection '{collection_name}'. "
        f"Duration: {total_duration:.2f}s. "
        f"Added: {added_count}, Skipped (already existed): {skipped_count}, Errors: {error_count}"
    )
    return added_count, skipped_count, error_count


# --- Main Script Logic ---

# def main(config_path: str = "config.json"):
def main(config_path: str = "config_fast.json"):
    """
    Main function to create ChromaDB collections for specified languages,
    chunk sizes, and overlap sizes, using parameters from the provided config file path.
    """
    print("--- Starting Batch ChromaDB Collection Creation Script ---")
    # Resolve the config path relative to the project root for consistency
    absolute_config_path = project_root / config_path
    print(f"Using configuration file: {absolute_config_path}")


    # --- Load Configuration ---
    try:
        # Use the absolute path to load config
        config_loader = ConfigLoader(str(absolute_config_path))
        config = config_loader.config
        language_configs = config.get("language_configs", [])
        rag_params = config.get("rag_parameters", {})

        # Load chunk and overlap sizes directly from config's rag_parameters
        chunk_sizes_to_create = rag_params.get("chunk_sizes_to_test")
        overlap_sizes_to_create = rag_params.get("overlap_sizes_to_test")

        # Validate that the parameters were found and are lists
        if chunk_sizes_to_create is None:
            raise ValueError("Error: 'chunk_sizes_to_test' not found in 'rag_parameters' section of config.")
        if not isinstance(chunk_sizes_to_create, list):
             raise ValueError("Error: 'chunk_sizes_to_test' in config must be a list.")
        if overlap_sizes_to_create is None:
            raise ValueError("Error: 'overlap_sizes_to_test' not found in 'rag_parameters' section of config.")
        if not isinstance(overlap_sizes_to_create, list):
             raise ValueError("Error: 'overlap_sizes_to_test' in config must be a list.")

        print(f"Target Chunk Sizes from config: {chunk_sizes_to_create}")
        print(f"Target Overlap Sizes from config: {overlap_sizes_to_create}")

        if not language_configs:
            print(f"Warning: No 'language_configs' found in '{absolute_config_path}'. No language-specific collections will be processed.")
            # Decide if we should exit or continue (maybe there are non-language specific tasks?)
            # For now, let's allow continuing, but log clearly.
            # return # Or uncomment to exit if languages are mandatory

    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{absolute_config_path}'. Exiting.")
        return
    except ValueError as ve: # Catch specific validation errors
        print(ve)
        return
    except Exception as e:
        print(f"Error loading or parsing configuration from '{absolute_config_path}': {e}. Exiting.")
        return

    # --- Initialize Shared Components ---
    try:
        print("Initializing ChromaDB client...")
        # Use absolute path for persistence directory as well
        absolute_persist_dir = project_root / PERSIST_DIRECTORY
        chroma_client = chromadb.PersistentClient(path=str(absolute_persist_dir))
        print(f"ChromaDB client initialized. Persistence directory: '{absolute_persist_dir}'")

        print("Initializing Embedding Retriever (for tokenizer and vectorization)...")
        # TODO: Consider making the embedding model configurable via config file for Retriever
        retriever = EmbeddingRetriever()
        if not hasattr(retriever, 'tokenizer') or not hasattr(retriever, 'vectorize_text'):
             raise AttributeError("Retriever must have 'tokenizer' and 'vectorize_text' methods.")
        print("Embedding Retriever initialized.")

        print("Initializing SentenceTransformer Embedding Function for ChromaDB...")
        # Get embedding model from config, fallback to a default if not specified
        # Ensure this model is compatible with the one potentially used by EmbeddingRetriever
        embedding_model_name = config.get("embedding_model_name", "Alibaba-NLP/gte-Qwen2-7B-instruct")
        print(f"Using embedding model for Chroma: {embedding_model_name}")
        gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        print("SentenceTransformer Embedding Function initialized.")

    except Exception as e:
        print(f"Error during initialization of ChromaDB/Retriever/Embedding Function: {e}. Exiting.")
        return

    # --- Create Output Directory if it doesn't exist ---
    # Use absolute path
    absolute_persist_dir.mkdir(parents=True, exist_ok=True)


    # --- Loop Through Languages and Parameters ---
    total_collections_processed = 0
    total_collections_created = 0
    total_collections_skipped = 0

    if not language_configs:
        print("No language configurations found to process.")

    for lang_config in language_configs:
        language = lang_config.get("language")
        manual_path_relative = lang_config.get("manual_path") # Path relative to project root
        base_collection_name = lang_config.get("collection_base_name")

        if not all([language, manual_path_relative, base_collection_name]):
            print(f"Warning: Skipping invalid language config entry: {lang_config}")
            continue

        # Resolve manual path relative to project root
        absolute_manual_path = project_root / manual_path_relative

        print(f"\n===== Processing Language: {language.upper()} (Manual: {absolute_manual_path}) =====")

        # --- Load Manual Text Once Per Language ---
        try:
            print(f"  Loading manual text from: {absolute_manual_path}")
            with open(absolute_manual_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"  Manual text loaded ({len(text)} characters).")
        except FileNotFoundError:
            print(f"  !!! ERROR: Manual file not found at '{absolute_manual_path}'. Skipping language '{language}'.")
            continue
        except Exception as e:
            print(f"  !!! ERROR reading manual file '{absolute_manual_path}': {e}. Skipping language '{language}'.")
            continue

        # --- Iterate Through Parameter Combinations from Config ---
        for chunk_size in chunk_sizes_to_create:
            for overlap_size in overlap_sizes_to_create:

                # Validate chunk/overlap combination
                if not isinstance(chunk_size, int) or chunk_size <= 0:
                     print(f"  Skipping invalid chunk_size: {chunk_size}")
                     continue
                if not isinstance(overlap_size, int) or overlap_size < 0:
                     print(f"  Skipping invalid overlap_size: {overlap_size}")
                     continue
                if overlap_size >= chunk_size:
                    print(f"  Skipping combination cs={chunk_size}, os={overlap_size} (overlap >= chunk size).")
                    continue

                total_collections_processed += 1
                dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
                print(f"\n--- Checking Collection: '{dynamic_collection_name}' (cs={chunk_size}, os={overlap_size}) ---")

                # --- Check if Collection Exists ---
                collection_exists = False
                collection = None
                try:
                    collection = chroma_client.get_collection(
                        name=dynamic_collection_name,
                        embedding_function=gte_embedding_function
                    )
                    print(f"  Collection '{dynamic_collection_name}' already exists. Skipping creation.")
                    collection_exists = True
                    total_collections_skipped += 1
                except Exception as e:
                    # Assuming exception means it doesn't exist. More specific handling is possible.
                    print(f"  Collection '{dynamic_collection_name}' does not seem to exist or error checking: {e}. Proceeding with creation attempt.")
                    collection_exists = False

                # --- Create and Populate if Not Exists ---
                if not collection_exists:
                    try:
                        print(f"  Creating new collection: '{dynamic_collection_name}'")
                        collection = chroma_client.create_collection(
                            name=dynamic_collection_name,
                            embedding_function=gte_embedding_function
                        )

                        print(f"  Splitting text using token limits: chunk_size={chunk_size}, overlap={overlap_size}")
                        if not retriever.tokenizer:
                             raise ValueError("Tokenizer not available in retriever, cannot split by token.")

                        token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                            tokenizer=retriever.tokenizer,
                            chunk_size=chunk_size,
                            chunk_overlap=overlap_size,
                        )
                        document_chunks_text = token_splitter.split_text(text)
                        print(f"  Generated {len(document_chunks_text)} token-based chunks for '{dynamic_collection_name}'.")

                        if not document_chunks_text:
                            print(f"  Warning: No chunks generated for {dynamic_collection_name}. Collection will be empty.")
                            total_collections_created += 1
                            continue

                        # --- Embed and Add Chunks ---
                        added, skipped, errors = embed_and_add_chunks_to_db(document_chunks_text, collection, retriever)
                        if errors == 0 and added > 0:
                             total_collections_created += 1
                             print(f"  Successfully created and populated collection '{dynamic_collection_name}'.")
                        elif errors > 0:
                             print(f"  !!! ERROR: Collection '{dynamic_collection_name}' created but encountered {errors} errors during embedding.")
                        elif added == 0 and skipped > 0:
                             total_collections_created += 1
                             print(f"  Collection '{dynamic_collection_name}' created, but all {skipped} chunks already existed (potentially from a prior interrupted run).")
                        else: # added == 0 and skipped == 0 and errors == 0
                             total_collections_created += 1
                             print(f"  Collection '{dynamic_collection_name}' created, but no chunks were added or skipped.")


                    except Exception as e:
                        import traceback
                        print(f"  !!! FATAL ERROR creating or populating collection '{dynamic_collection_name}': {e}")
                        print(traceback.format_exc())
                        try:
                            print(f"  Attempting to clean up potentially failed collection '{dynamic_collection_name}'...")
                            chroma_client.delete_collection(name=dynamic_collection_name)
                            print(f"  Cleanup successful.")
                        except Exception as delete_e:
                            print(f"  !!! Warning: Failed to cleanup collection '{dynamic_collection_name}' after error: {delete_e}")
                        continue

    print("\n--- Batch Creation Summary ---")
    print(f"Total parameter combinations checked: {total_collections_processed}")
    print(f"Total collections successfully created/populated in this run: {total_collections_created}")
    print(f"Total collections skipped (already existed): {total_collections_skipped}")
    print("--- Batch ChromaDB Collection Creation Script Finished ---")


if __name__ == "__main__":
    # Allows running the script directly, using the default config path relative to the project root
    # If running from the 'utils' directory, the path adjustment at the top handles finding modules.
    # The config path is now resolved relative to the project root inside main().
    main()

# Fix the error, so that it doesn't happen when main.py is run
# Also ensure chunk/overlap sizes are loaded from config