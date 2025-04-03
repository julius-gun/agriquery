# RAG/rag_tester.py
import json
import os
import time
import datetime
import hashlib # Added for generating unique keys

# Import analysis tools and the specific metrics calculation function
from analysis.analysis_tools import analyze_dataset_across_types, load_dataset, analyze_evaluation_results
from evaluation.metrics import calculate_metrics # Import the core function
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from parameter_tuning.parameters import RagParameters
# Assuming collection and initialize_retriever are needed from rag_pipeline
# If rag_pipeline.py handles DB setup separately, these might not be needed here directly
# For now, assume they are available or initialized elsewhere as needed by the retriever.
# We might need to adjust imports based on final structure. Let's assume retriever setup happens before this script.
from rag_pipeline import initialize_retriever, collection
from utils.config_loader import ConfigLoader

# --- Helper Functions for Result File Handling ---

def load_or_initialize_results(filepath: str) -> dict:
    """Loads results from a JSON file or returns an empty dict if not found/invalid."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {filepath}. Starting with empty results.")
            return {}
        except Exception as e:
            print(f"Warning: Error loading results file {filepath}: {e}. Starting with empty results.")
            return {}
    else:
        return {}

def save_results(filepath: str, data: dict):
    """Saves the results data to a JSON file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error: Failed to save results to {filepath}: {e}")

def generate_question_key(dataset_name: str, question: str) -> str:
    """Generates a unique key for a question within a dataset."""
    # Using SHA256 hash for a consistent and filesystem-friendly key
    return hashlib.sha256(f"{dataset_name}_{question}".encode()).hexdigest()

# --- Main Test Function ---

def run_rag_test(config_path="config.json"):
    """
    Runs the RAG test based on the provided configuration, calculating
    overall metrics across all datasets.
    """
    config_loader = ConfigLoader(config_path)
    rag_config = config_loader.config # Load the entire config

    # --- Load Models ---
    # Pass the entire llm_models dictionary to the manager
    llm_connector_manager = LLMConnectorManager(rag_config["llm_models"])
    question_model_name = config_loader.get_question_model_name()
    # Determine LLM type for question model (assuming ollama, adjust if needed)
    # TODO: Make LLM type configurable per model if necessary
    question_llm_type = "ollama"
    question_llm_connector = llm_connector_manager.get_connector(question_llm_type, question_model_name)

    evaluator_model_name = config_loader.get_evaluator_model_name()
    # Determine LLM type for evaluator model (assuming ollama, adjust if needed)
    # TODO: Make LLM type configurable per model if necessary
    evaluator_llm_type = "ollama"
    evaluator_llm_connector = llm_connector_manager.get_connector(evaluator_llm_type, evaluator_model_name)

    # --- Load Prompts ---
    question_prompt_template = config_loader.load_prompt_template("question_prompt")
    evaluation_prompt_template = config_loader.load_prompt_template("evaluation_prompt")

    # --- Initialize Evaluator ---
    evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template)

    # --- Load Datasets ---
    dataset_paths = config_loader.get_question_dataset_paths()

    # --- Initialize Lists/Dicts for Aggregation ---
    all_individual_results = [] # List to store results from ALL datasets for overall metrics
    per_dataset_details = {} # Dict to store results grouped by dataset for saving

    overall_start_time = time.time()

    # --- Process Each Dataset ---
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        dataset = load_dataset(dataset_path)
        if not dataset:
            print(f"Skipping dataset: {dataset_name} due to loading error.")
            continue

        dataset_specific_results = [] # Results for the current dataset only
        dataset_start_time = time.time()

        for i, question_data in enumerate(dataset):
            question = question_data.get("question")
            expected_answer = question_data.get("answer")
            page = question_data.get("page", "N/A") # Default page to N/A if missing

            if not question or expected_answer is None: # Check if essential data is present
                print(f"Warning: Skipping entry {i} in {dataset_name} due to missing question or answer.")
                continue

            print(f"Processing Q{i+1}/{len(dataset)}: {question[:80]}...") # Progress indicator

            # --- RAG Retrieval ---
            try:
                rag_params_dict = config_loader.get_rag_parameters()
                rag_params = RagParameters.from_dict(rag_params_dict)
                retriever = initialize_retriever(rag_params.retrieval_algorithm) # Initialize retriever inside loop if params can change per dataset? Or outside if static. Assuming static for now.
                question_embedding = retriever.vectorize_text(question)

                # Fetch embeddings and documents - consider optimizing if performance is an issue
                # Ensure 'collection' is initialized correctly (e.g., from rag_pipeline.py or here)
                db_results = collection.get(include=['embeddings', 'documents']) # Renamed to avoid conflict
                document_chunk_embeddings_from_db = db_results.get('embeddings')
                document_chunks_text_from_db = db_results.get('documents')

                if document_chunk_embeddings_from_db is None or document_chunks_text_from_db is None:
                     print(f"Warning: Could not retrieve embeddings or documents from DB for question: {question}")
                     context = "Error: Could not retrieve context from database."
                     retrieved_chunks_text = []
                else:
                    retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(
                        question_embedding,
                        document_chunk_embeddings_from_db,
                        document_chunks_text_from_db,
                        top_k=rag_params.num_retrieved_docs
                    )
                    context = "\n".join(retrieved_chunks_text)

            except Exception as e:
                print(f"Error during RAG retrieval for question '{question}': {e}")
                context = "Error during retrieval."
                retrieved_chunks_text = []
                model_answer = "Error: Failed during retrieval."
                evaluation_result = "error" # Mark as error

            # --- LLM Question Answering (only if retrieval didn't fail hard) ---
            if evaluation_result != "error": # Proceed only if retrieval was okay
                try:
                    prompt = question_prompt_template.format(context=context, question=question)
                    model_answer = question_llm_connector.invoke(prompt)
                except Exception as e:
                    print(f"Error during LLM QA invocation for question '{question}': {e}")
                    model_answer = "Error: Failed during QA generation."
                    evaluation_result = "error" # Mark as error

            # --- Evaluation (only if QA didn't fail) ---
            if evaluation_result != "error":
                try:
                    # Ensure expected_answer is a string for the evaluator
                    evaluation_result = evaluator.evaluate_answer(question, model_answer, str(expected_answer))
                    print(f"  Evaluator judgment: {evaluation_result}") # Debug print
                except Exception as e:
                    print(f"Error during evaluation for question '{question}': {e}")
                    evaluation_result = "error" # Mark evaluation as error

            # --- Store Result ---
            # Ensure the keys match what calculate_metrics expects: 'self_evaluation' and 'dataset'
            result_entry = {
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "retrieved_context": context, # Optionally store context
                "self_evaluation": evaluation_result.strip().lower() if isinstance(evaluation_result, str) else "error", # Store evaluator result, handle errors
                "page": page,
                "dataset": dataset_name, # Store the source dataset name/key
            }
            dataset_specific_results.append(result_entry)
            all_individual_results.append(result_entry) # Add to the overall list

        dataset_end_time = time.time()
        dataset_duration = dataset_end_time - dataset_start_time
        print(f"Finished dataset {dataset_name} in {dataset_duration:.2f} seconds.")

        # Store dataset-specific details for saving later
        per_dataset_details[dataset_name] = {
            "results": dataset_specific_results,
            "duration": dataset_duration,
            "total_questions_processed": len(dataset_specific_results)
        }

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    print(f"\n--- Finished processing all datasets in {overall_duration:.2f} seconds ---")

    # --- Calculate Overall Metrics (after processing all datasets) ---
    print("\n--- Calculating Overall Metrics ---")
    if not all_individual_results:
        print("No results collected. Cannot calculate overall metrics.")
        overall_metrics = {} # Assign empty dict if no results
    else:
        # Pass the aggregated list to the updated calculate_metrics function
        overall_metrics = calculate_metrics(all_individual_results)

        print("\n--- Overall Evaluation Analysis (All Datasets) ---")
        # Print all calculated metrics
        for key, value in overall_metrics.items():
             if isinstance(value, float):
                 print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
             else:
                 print(f"  {key.replace('_', ' ').title()}: {value}")

        # You can still use analyze_evaluation_results if you only want the subset it prints
        # analyze_evaluation_results(overall_metrics, "Overall (All Datasets)")


    # --- Overall Dataset Analysis (Counts per type) ---
    analyze_dataset_across_types(dataset_paths) # Analyze counts across datasets

    # --- Save Results to JSON ---
    output_dir = config_loader.get_output_dir() # Use getter for output dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Use more robust datetime for timestamp
    results_filename = os.path.join(output_dir, f"rag_test_results_{timestamp}.json")

    # Structure the final JSON output
    final_results_to_save = {
        "test_timestamp": timestamp,
        "question_model": question_model_name,
        "evaluator_model": evaluator_model_name,
        "rag_parameters": config_loader.get_rag_parameters(),
        "overall_metrics": overall_metrics, # Include the calculated overall metrics
        "overall_duration_seconds": overall_duration,
        "per_dataset_details": per_dataset_details # Include the detailed results per dataset
    }

    save_results(results_filename, final_results_to_save)
    print(f"\nResults saved to: {results_filename}")


if __name__ == "__main__":
    # run_rag_test() # Example using default config.json
    run_rag_test("config_fast.json") # Example of running with a specific config file