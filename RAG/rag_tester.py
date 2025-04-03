# RAG/testers/rag_tester.py
import json
import os
import time
import datetime
import hashlib # Added for generating unique keys

from analysis.analysis_tools import analyze_dataset_across_types, load_dataset, calculate_and_analyze_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from parameter_tuning.parameters import RagParameters
# Assuming collection and initialize_retriever are needed from rag_pipeline
# If rag_pipeline.py handles DB setup separately, these might not be needed here directly
# For now, assume they are available or initialized elsewhere as needed by the retriever.
# We might need to adjust imports based on final structure. Let's assume retriever setup happens before this script.
from rag_pipeline import initialize_retriever, collection
from utils.config_loader import ConfigLoader
# Note: calculate_metrics is used within calculate_and_analyze_metrics, direct import might not be needed here.

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
    Runs the RAG test based on the provided configuration.
    """
    config_loader = ConfigLoader(config_path)
    rag_config = config_loader.config # Load the entire config

    # --- Load Models ---
    # Pass the entire llm_models dictionary to the manager
    llm_connector_manager = LLMConnectorManager(rag_config["llm_models"])
    question_model_name = config_loader.get_question_model_name() # Use getter
    question_llm_connector = llm_connector_manager.get_connector("ollama", question_model_name) # Assuming ollama for now

    evaluator_model_name = config_loader.get_evaluator_model_name()
    evaluator_llm_connector = llm_connector_manager.get_connector("ollama", evaluator_model_name) # Assuming ollama for now


    # --- Load Prompts ---
    question_prompt_template = config_loader.load_prompt_template("question_prompt")
    evaluation_prompt_template = config_loader.load_prompt_template("evaluation_prompt")

    # --- Initialize Evaluator ---
    evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template)

    # --- Load Datasets ---
    dataset_paths = config_loader.get_question_dataset_paths() # Use getter for dataset paths
    all_results = {}

    for dataset_name, dataset_path in dataset_paths.items():
        dataset = load_dataset(dataset_path)
        if not dataset:
            print(f"Skipping dataset: {dataset_name} due to loading error.")
            continue

        dataset_results = []
        start_time = time.time()

        for question_data in dataset:
            question = question_data["question"]
            expected_answer = question_data["answer"]
            page = question_data["page"] # Assuming page number is available

            # --- RAG Retrieval ---
            rag_params_dict = config_loader.get_rag_parameters() # Get RAG parameters dictionary
            rag_params = RagParameters.from_dict(rag_params_dict) # Create RagParameters object

            # Use attribute access for RagParameters object
            retriever = initialize_retriever(rag_params.retrieval_algorithm)
            question_embedding = retriever.vectorize_text(question)

            # Fetch embeddings and documents - consider optimizing if performance is an issue
            results = collection.get(include=['embeddings', 'documents'])
            document_chunk_embeddings_from_db = results['embeddings']
            document_chunks_text_from_db = results['documents']

            # Use attribute access for RagParameters object
            retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(
                question_embedding,
                document_chunk_embeddings_from_db,
                document_chunks_text_from_db,
                top_k=rag_params.num_retrieved_docs
            )
            context = "\n".join(retrieved_chunks_text) # Concatenate retrieved chunks into context string


            # --- LLM Question Answering ---
            prompt = question_prompt_template.format(context=context, question=question)
            model_answer = question_llm_connector.invoke(prompt)


            # --- Evaluation ---
            evaluation_result = evaluator.evaluate_answer(question, model_answer, expected_answer)
            print(f"Evaluation Result: {evaluation_result}") # Debug print

            result_entry = {
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "self_evaluation": evaluation_result, # Changed 'evaluation' to 'self_evaluation' and store evaluator result
                "page": page,
                "dataset": dataset_name,
            }
            print(f"Dataset Name: {dataset_name}") # Debug print
            print(f"Result Entry: {result_entry}") # Debug print
            dataset_results.append(result_entry)

        end_time = time.time()
        dataset_duration = end_time - start_time

        print(f"Dataset Results before metrics calculation: {dataset_results}") # Debug print
        evaluation_metrics = calculate_and_analyze_metrics(dataset_results, dataset_name) # Calculate and analyze metrics

        dataset_evaluation_results = { # Use the metrics dictionary directly
            "metrics": evaluation_metrics,
            "total_questions": len(dataset_results),
            "duration": dataset_duration
        }


        all_results[dataset_name] = {
            "results": dataset_results,
            "evaluation_summary": dataset_evaluation_results
        }
        analyze_evaluation_results(dataset_evaluation_results["metrics"], dataset_name) # Pass metrics to analysis function


    # --- Overall Analysis ---
    analyze_dataset_across_types(dataset_paths) # Analyze across datasets
    print("\n--- Overall RAG Test Analysis ---")
    for dataset_name, result_data in all_results.items():
        summary = result_data["evaluation_summary"]
        metrics = summary["metrics"] # Get metrics from summary
        print(f"Dataset: {dataset_name}, Accuracy: {metrics['accuracy']:.2f}%, Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%, F1 Score: {metrics['f1_score']:.2f}%, Duration: {summary['duration']:.2f}s")


    # --- Save Results to JSON ---
    output_dir = config_loader.get_output_dir() # Use getter for output dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Use more robust datetime for timestamp
    results_filename = os.path.join(output_dir, f"rag_test_results_{timestamp}.json")
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to: {results_filename}")


if __name__ == "__main__":
    # run_rag_test()
    run_rag_test("config_fast.json") # Example of running with a specific config file