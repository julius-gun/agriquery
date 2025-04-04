# create_databases.py
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import os
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming these modules are accessible from the project root
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from utils.config_loader import ConfigLoader

# --- Configuration ---
PERSIST_DIRECTORY = "chroma_db"
CONFIG_FILE_PATH = "config.json"

# --- Parameters for Database Creation ---
# Define the specific chunk and overlap sizes you want to generate databases for
CHUNK_SIZES_TO_CREATE = [200, 500, 1000, 2000]
OVERLAP_SIZES_TO_CREATE = [100, 200, 500]

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

    # Batch processing can be much faster if the collection supports it well
    # For simplicity here, we process one by one, but consider batching for large datasets
    ids_to_add = []
    embeddings_to_add = []
    documents_to_add = []

    # First, identify which chunks need embedding (check existence)
    existing_ids = set(collection.get(include=[])['ids']) # Get all existing IDs efficiently

    for i, chunk_text in enumerate(document_chunks_text):
        chunk_id = generate_chunk_id(chunk_text)
        if chunk_id in existing_ids:
            skipped_count += 1
        else:
            try:
                # Vectorize only if not skipped
                chunk_embedding = retriever.vectorize_text(chunk_text)
                ids_to_add.append(chunk_id)
                embeddings_to_add.append(chunk_embedding[0]) # vectorize_text returns a list containing one embedding
                documents_to_add.append(chunk_text)
                added_count += 1
                existing_ids.add(chunk_id) # Add to set to prevent duplicate adds in this run
            except Exception as e:
                print(f"  !!! ERROR vectorizing chunk {i} for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
                error_count += 1

        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == total_chunks:
             current_time = time.time()
             elapsed = current_time - start_time
             print(f"    Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks... "
                   f"(Checked: {skipped_count+added_count+error_count}, To Add: {added_count}, Errors: {error_count}) "
                   f"Elapsed: {elapsed:.2f}s")

    # Add collected chunks in batches (or potentially one large batch)
    if ids_to_add:
        try:
            print(f"  Adding {len(ids_to_add)} new chunks to collection '{collection_name}'...")
            # Add in batches to avoid potential limits (e.g., batch size of 1000)
            batch_size = 1000
            for i in range(0, len(ids_to_add), batch_size):
                batch_ids = ids_to_add[i:i+batch_size]
                batch_embeddings = embeddings_to_add[i:i+batch_size]
                batch_documents = documents_to_add[i:i+batch_size]
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents
                )
                print(f"    Added batch {i//batch_size + 1}/{(len(ids_to_add) + batch_size - 1)//batch_size}")
            print(f"  Successfully added {len(ids_to_add)} chunks.")
        except Exception as e:
            print(f"  !!! ERROR adding batch to collection '{collection_name}': {e}")
            # Note: This might leave the collection partially updated
            error_count += len(ids_to_add) # Count all potential adds as errors if batch fails
            added_count = 0 # Reset added count as the batch failed

    end_time = time.time()
    total_duration = end_time - start_time
    print(
        f"  Embedding finished for collection '{collection_name}'. "
        f"Duration: {total_duration:.2f}s. "
        f"Added: {added_count}, Skipped (already existed): {skipped_count}, Errors: {error_count}"
    )
    return added_count, skipped_count, error_count


# --- Main Script Logic ---

def main():
    """
    Main function to create ChromaDB collections for specified languages,
    chunk sizes, and overlap sizes.
    """
    print("--- Starting Batch ChromaDB Collection Creation Script ---")

    # --- Load Configuration ---
    try:
        config_loader = ConfigLoader(CONFIG_FILE_PATH)
        language_configs = config_loader.config.get("language_configs", [])
        if not language_configs:
            print(f"Error: No 'language_configs' found in '{CONFIG_FILE_PATH}'. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{CONFIG_FILE_PATH}'. Exiting.")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}. Exiting.")
        return

    # --- Initialize Shared Components ---
    try:
        print("Initializing ChromaDB client...")
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        print(f"ChromaDB client initialized. Persistence directory: '{PERSIST_DIRECTORY}'")

        print("Initializing Embedding Retriever (for tokenizer and vectorization)...")
        # Assuming EmbeddingRetriever uses the model specified in its definition
        # If model needs to be configurable, adjust EmbeddingRetriever or pass params
        retriever = EmbeddingRetriever()
        if not hasattr(retriever, 'tokenizer') or not hasattr(retriever, 'vectorize_text'):
             raise AttributeError("Retriever must have 'tokenizer' and 'vectorize_text' methods.")
        print("Embedding Retriever initialized.")

        print("Initializing SentenceTransformer Embedding Function for ChromaDB...")
        # Ensure this model matches or is compatible with the one used in EmbeddingRetriever
        # Using the same model name as likely used in EmbeddingRetriever init
        gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
        )
        print("SentenceTransformer Embedding Function initialized.")

    except Exception as e:
        print(f"Error during initialization: {e}. Exiting.")
        return

    # --- Create Output Directory if it doesn't exist ---
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # --- Loop Through Languages and Parameters ---
    total_collections_processed = 0
    total_collections_created = 0
    total_collections_skipped = 0

    for lang_config in language_configs:
        language = lang_config.get("language")
        manual_path = lang_config.get("manual_path")
        base_collection_name = lang_config.get("collection_base_name")

        if not all([language, manual_path, base_collection_name]):
            print(f"Warning: Skipping invalid language config entry: {lang_config}")
            continue

        print(f"\n===== Processing Language: {language.upper()} (Manual: {manual_path}) =====")

        # --- Load Manual Text Once Per Language ---
        try:
            print(f"  Loading manual text from: {manual_path}")
            with open(manual_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"  Manual text loaded ({len(text)} characters).")
        except FileNotFoundError:
            print(f"  !!! ERROR: Manual file not found at '{manual_path}'. Skipping language '{language}'.")
            continue
        except Exception as e:
            print(f"  !!! ERROR reading manual file '{manual_path}': {e}. Skipping language '{language}'.")
            continue

        # --- Iterate Through Parameter Combinations ---
        for chunk_size in CHUNK_SIZES_TO_CREATE:
            for overlap_size in OVERLAP_SIZES_TO_CREATE:

                # Basic validation: Overlap should not be larger than chunk size
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
                    # Try to get the collection. If it succeeds, it exists.
                    collection = chroma_client.get_collection(
                        name=dynamic_collection_name,
                        embedding_function=gte_embedding_function # Good practice to provide EF
                    )
                    print(f"  Collection '{dynamic_collection_name}' already exists. Skipping creation.")
                    collection_exists = True
                    total_collections_skipped += 1
                except Exception as e:
                    # Crude check, specific exception handling (e.g., for ValueError if not found) is better
                    # print(f"  Debug: Exception during get_collection: {type(e).__name__} - {e}") # Optional debug
                    print(f"  Collection '{dynamic_collection_name}' does not seem to exist. Proceeding with creation.")
                    collection_exists = False

                # --- Create and Populate if Not Exists ---
                if not collection_exists:
                    try:
                        print(f"  Creating new collection: '{dynamic_collection_name}'")
                        collection = chroma_client.create_collection(
                            name=dynamic_collection_name,
                            embedding_function=gte_embedding_function
                            # metadata={"hnsw:space": "cosine"} # Optional: Specify distance metric if needed
                        )

                        print(f"  Splitting text using token limits: chunk_size={chunk_size}, overlap={overlap_size}")
                        token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                            tokenizer=retriever.tokenizer,
                            chunk_size=chunk_size,
                            chunk_overlap=overlap_size,
                            # length_function=len # Use character length if tokenizer unavailable, but token is preferred
                        )
                        document_chunks_text = token_splitter.split_text(text)
                        print(f"  Generated {len(document_chunks_text)} token-based chunks for '{dynamic_collection_name}'.")

                        if not document_chunks_text:
                            print(f"  Warning: No chunks generated for {dynamic_collection_name}. Collection will be empty.")
                            total_collections_created += 1 # Count as created, even if empty
                            continue # Skip embedding if no chunks

                        # --- Embed and Add Chunks ---
                        added, skipped, errors = embed_and_add_chunks_to_db(document_chunks_text, collection, retriever)
                        if errors == 0 and added > 0:
                             total_collections_created += 1
                             print(f"  Successfully created and populated collection '{dynamic_collection_name}'.")
                        elif errors > 0:
                             print(f"  !!! ERROR: Collection '{dynamic_collection_name}' created but encountered {errors} errors during embedding.")
                             # Decide if partially created collection should be deleted? For now, leave it.
                        elif added == 0 and skipped > 0:
                             total_collections_created += 1 # Created, but all chunks existed from a previous partial run?
                             print(f"  Collection '{dynamic_collection_name}' created, but all {skipped} chunks already existed (potentially from a prior interrupted run).")
                        else: # added == 0 and skipped == 0 and errors == 0 (empty text?)
                             total_collections_created += 1
                             print(f"  Collection '{dynamic_collection_name}' created, but no chunks were added or skipped (was the input text empty?).")


                    except Exception as e:
                        print(f"  !!! FATAL ERROR creating or populating collection '{dynamic_collection_name}': {e}")
                        # Attempt to delete potentially partially created collection on fatal error?
                        try:
                            print(f"  Attempting to clean up potentially failed collection '{dynamic_collection_name}'...")
                            chroma_client.delete_collection(name=dynamic_collection_name)
                            print(f"  Cleanup successful.")
                        except Exception as delete_e:
                            print(f"  !!! Warning: Failed to cleanup collection '{dynamic_collection_name}' after error: {delete_e}")
                        # Continue to next iteration
                        continue

    print("\n--- Batch Creation Summary ---")
    print(f"Total parameter combinations checked: {total_collections_processed}")
    print(f"Total collections successfully created/populated in this run: {total_collections_created}")
    print(f"Total collections skipped (already existed): {total_collections_skipped}")
    print("--- Batch ChromaDB Collection Creation Script Finished ---")


if __name__ == "__main__":
    main()