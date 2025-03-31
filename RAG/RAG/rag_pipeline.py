import chromadb
from chromadb.utils import embedding_functions
import hashlib
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
from evaluation.evaluation_metrics import evaluate_rag_pipeline
from analysis.analysis_tools import analyze_evaluation_results, load_dataset, analyze_dataset_across_types
import json
import os

# --- Configuration ---
persist_directory = 'chroma_db'
collection_name = "english_manual"
path_to_text = 'english_manual.txt'

# --- Dataset paths are now defined here, can be moved to a config file later ---
dataset_paths = {
    "general_questions": 'RAG\question_datasets\question_answers_pairs.json',
    "table_questions": 'RAG\question_datasets\question_answers_tables.json',
    "unanswerable_questions": 'RAG\question_datasets\question_answers_unanswerable.json'
}
config_file_path = 'RAG/config.json' # Path to config file

# Load configuration from JSON file
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
else:
    raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")
rag_params = config.get("rag_parameters", {}) # Load RAG parameters

# --- Initialize Components ---
gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="Alibaba-NLP/gte-Qwen2-7B-instruct")
chroma_client = chromadb.PersistentClient(path=persist_directory)

try:
    collection = chroma_client.get_collection(name=collection_name, embedding_function=gte_embedding_function)
    print(f"Collection '{collection_name}' loaded successfully.")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(name=collection_name, embedding_function=gte_embedding_function)
    print(f"Collection '{collection_name}' created.")

def initialize_retriever(retrieval_strategy_str: str):
    """Initializes the retriever based on the specified strategy."""
    if retrieval_strategy_str == "embedding":
        return EmbeddingRetriever()
    elif retrieval_strategy_str == "keyword":
        return KeywordRetriever() # Instantiate and return KeywordRetriever
    elif retrieval_strategy_str == "hybrid":
        raise NotImplementedError("Hybrid retrieval not yet implemented.")  # Placeholder for future
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy_str}")

retriever = initialize_retriever(rag_params.get("retrieval_algorithm", "embedding")) # Initialize retriever based on config, default to embedding
tokenizer = retriever.tokenizer if isinstance(retriever, EmbeddingRetriever) else None # Conditionally set tokenizer

# --- Document Loading and Chunking ---
def chunk_text_with_overlap(text, tokenizer=None, chunk_size=1000, overlap_size=100): # Make tokenizer optional and default to None
    """Chunks text into token-based chunks with overlap."""
    chunks = []
    if tokenizer: # Check if tokenizer is available before using it
        tokens = tokenizer.tokenize(text) # Tokenize the entire text
        start_index = 0
        while start_index < len(tokens):
            end_index = min(start_index + chunk_size, len(tokens)) # token-based chunking
            chunk_tokens = tokens[start_index:end_index]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens) # Convert tokens back to text
            chunks.append(chunk_text)
            start_index += chunk_size - overlap_size # Move start index by chunk size minus overlap
    else: # Fallback to character-based chunking if no tokenizer
        start_index = 0
        text_length = len(text)
        while start_index < text_length:
            end_index = min(start_index + chunk_size, text_length)
            chunk_text = text[start_index:end_index]
            chunks.append(chunk_text)
            start_index += chunk_size - overlap_size
    return chunks


with open(path_to_text, 'r') as f:
    text = f.read() # Read text content
document_chunks_text = chunk_text_with_overlap(text, tokenizer, rag_params.get("chunk_size", 1000), rag_params.get("overlap_size", 20)) # Pass tokenizer to chunking function


# --- Vectorization and Add to ChromaDB ---
def generate_chunk_id(text_chunk, chunk_size): # Modified: Include chunk_size in ID
    """Generates a unique ID for a text chunk using SHA256 hashing and chunk size."""
    chunk_id_content = f"{text_chunk}-{chunk_size}" # Include chunk_size in the ID content
    return hashlib.sha256(chunk_id_content.encode()).hexdigest()

def embed_and_add_chunks_to_db(document_chunks_text, collection, chunk_size): # Modified: Pass chunk_size
    """Embeds text chunks and adds them to ChromaDB, avoiding duplicates."""
    for chunk_text in document_chunks_text:
        chunk_id = generate_chunk_id(chunk_text, chunk_size) # Pass chunk_size to ID generation
        if collection.get(ids=[chunk_id])['ids']: # Check if chunk ID already exists
            print(f"Chunk with ID '{chunk_id}' already exists. Skipping embedding.")
        else:
            chunk_embedding = retriever.vectorize_text(chunk_text)
            collection.add(
                embeddings=chunk_embedding,
                documents=[chunk_text],
                ids=[chunk_id]
            )
            print(f"Chunk with ID '{chunk_id}' embedded and added to ChromaDB.")

# --- Embedding and Adding Chunks to DB (Uncomment to run embedding) ---
embed_and_add_chunks_to_db(document_chunks_text, collection, rag_params.get("chunk_size", 1000)) # Run embedding with chunk_size from config, default to 1000


# --- Load and Evaluate Datasets ---
all_evaluation_results = {}
for dataset_name, dataset_path in dataset_paths.items(): # Iterate through dataset_paths
    dataset = load_dataset(dataset_path)
    if dataset:
        print(f"\n--- Evaluating on {dataset_name} dataset ---")
        evaluation_results = evaluate_rag_pipeline(dataset, retriever, collection, rag_params)
        all_evaluation_results[dataset_name] = evaluation_results
        analyze_evaluation_results(evaluation_results, dataset_name) # Analyze results per dataset

# --- Analyze Datasets ---
analyze_dataset_across_types(dataset_paths) # Pass dataset_paths to analysis function

# --- Overall Analysis ---
print("\n--- Overall Evaluation Analysis ---")
# Example of overall analysis (can be extended)
for dataset_name, results in all_evaluation_results.items(): # Iterate through evaluation results
    if "retrieval_accuracy" in results:
        print(f"Dataset: {dataset_name}, Retrieval Accuracy: {results['retrieval_accuracy']:.2f}%")