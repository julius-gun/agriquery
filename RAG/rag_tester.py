# RAG/rag_tester.py
import json
import os
import time
import datetime
import hashlib
import chromadb # Import chromadb

from analysis.analysis_tools import analyze_dataset_across_types, load_dataset
from evaluation.metrics import calculate_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from parameter_tuning.parameters import RagParameters
# Assuming collection and initialize_retriever are needed from rag_pipeline
# If rag_pipeline.py handles DB setup separately, these might not be needed here directly
# For now, assume they are available or initialized elsewhere as needed by the retriever.
# We might need to adjust imports based on final structure. Let's assume retriever setup happens before this script.
from rag_pipeline import retriever, collection
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
    Runs the RAG test against the specific ChromaDB collection
    matching the current chunk_size and overlap_size in the config.    
    1. Answers all questions using the QA LLM.
    2. Evaluates all answers using the Evaluator LLM.
    3. Calculates overall metrics.
    """
    config_loader = ConfigLoader(config_path)
    rag_config = config_loader.config # Load the entire config

    # --- Load Models ---
    # Pass the entire llm_models dictionary to the manager
    llm_connector_manager = LLMConnectorManager(rag_config["llm_models"])
    question_model_name = config_loader.get_question_model_name()
    question_llm_type = "ollama" # TODO: Make configurable if needed
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

    # --- Get RAG Parameters and Determine Collection ---
    rag_params_dict = config_loader.get_rag_parameters()
    rag_params = RagParameters.from_dict(rag_params_dict) # Use RagParameters class if needed, or just dict

    chunk_size = rag_params_dict.get("chunk_size", 2000)
    overlap_size = rag_params_dict.get("overlap_size", 50)
    base_collection_name = "english_manual" # Must match the base name used in rag_pipeline.py
    dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
    print(f"Attempting to use ChromaDB collection: '{dynamic_collection_name}' (based on config parameters)")

    # --- Initialize ChromaDB Client and Get Specific Collection ---
    persist_directory = "chroma_db" # Define persist directory path
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    try:
        # Get the specific collection required for this test run
        collection = chroma_client.get_collection(name=dynamic_collection_name)
        print(f"Successfully connected to collection '{dynamic_collection_name}'.")
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! ERROR: Failed to get ChromaDB collection '{dynamic_collection_name}'.")
        print(f"!!! Please ensure you have run 'rag_pipeline.py' with:")
        print(f"!!!   'chunk_size': {chunk_size}")
        print(f"!!!   'overlap_size': {overlap_size}")
        print(f"!!! in '{config_path}' to create and embed this collection.")
        print(f"!!! Original error: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return # Exit the test function if the required DB doesn't exist

    # --- Initialize Data Structures ---
    # Stores results after QA phase, before evaluation phase
    intermediate_results_by_dataset = {}
    overall_qa_start_time = time.time()

    # --- Phase 1: Question Answering ---
    print("\n--- Phase 1: Answering all questions ---")
    total_questions_to_answer = 0
    for dataset_name, dataset_path in dataset_paths.items():
        dataset = load_dataset(dataset_path)
        if dataset:
            total_questions_to_answer += len(dataset)

    answered_questions_count = 0
    # The retriever instance is already initialized globally based on config
    # The collection instance is now dynamically loaded above

    for dataset_name, dataset_path in dataset_paths.items():
        print(f"\nProcessing Dataset for QA: {dataset_name}")
        dataset = load_dataset(dataset_path)
        if not dataset:
            print(f"Skipping dataset: {dataset_name} due to loading error.")
            continue

        dataset_intermediate_results = []
        dataset_start_time = time.time()

        for i, question_data in enumerate(dataset):
            answered_questions_count += 1
            question = question_data.get("question")
            expected_answer = question_data.get("answer")
            page = question_data.get("page", "N/A")

            if not question or expected_answer is None:
                print(f"Warning: Skipping entry {i} in {dataset_name} due to missing question or answer.")
                # Store minimal error info if needed, or just skip
                intermediate_entry = {
                    "question": question or "Missing Question",
                    "expected_answer": expected_answer,
                    "model_answer": "Error: Missing essential data",
                    "page": page,
                    "dataset": dataset_name,
                    "qa_error": True
                }
                dataset_intermediate_results.append(intermediate_entry)
                continue

            print(f"Answering Q {answered_questions_count}/{total_questions_to_answer} (Dataset: {dataset_name} {i+1}/{len(dataset)}): {question[:80]}...")

            context = ""
            model_answer = ""
            qa_error = False

            # --- RAG Retrieval ---
            try:
                # Use the imported retriever instance directly
                question_embedding = retriever.vectorize_text(question)
                # Use the imported collection instance directly
                # Fetch only embeddings and documents needed for retrieval logic
                # NOTE: retriever.retrieve_relevant_chunks expects embeddings and text lists
                # Getting ALL embeddings/docs from Chroma can be slow for large DBs.
                # A more efficient way is to use collection.query()
                query_results = collection.query(
                    query_embeddings=question_embedding,
                    n_results=rag_params.num_retrieved_docs, # Use num_retrieved_docs here
                    include=['documents'] # Only need documents for context
                )

                # Check if query_results is structured as expected (list of lists)
                if query_results and query_results.get('documents') and isinstance(query_results['documents'], list) and len(query_results['documents']) > 0:
                     retrieved_chunks_text = query_results['documents'][0] # Chroma returns [[doc1, doc2,...]]
                     context = "\n".join(retrieved_chunks_text)
                     if not context:
                          print("  Warning: Retrieval returned empty documents.")
                          # Decide if this is an error or just no context found
                          # context = "No relevant context found." # Or set qa_error = True
                else:
                     print(f"  Warning: Could not retrieve documents from DB collection '{dynamic_collection_name}'. Results: {query_results}")
                     context = "Error: Could not retrieve context from database."
                     model_answer = "Error: Failed during retrieval."
                     qa_error = True

                # --- Old retrieval logic (less efficient for large DBs) ---
                # db_results = collection.get(include=['embeddings', 'documents'])
                # document_chunk_embeddings_from_db = db_results.get('embeddings')
                # document_chunks_text_from_db = db_results.get('documents')
                # if document_chunk_embeddings_from_db is None or document_chunks_text_from_db is None:
                #      print(f"  Warning: Could not retrieve embeddings or documents from DB.")
                #      context = "Error: Could not retrieve context from database."
                #      model_answer = "Error: Failed during retrieval."
                #      qa_error = True
                # else:
                #     retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(
                #         question_embedding,
                #         document_chunk_embeddings_from_db, # This requires getting ALL embeddings
                #         document_chunks_text_from_db,      # This requires getting ALL documents
                #         top_k=rag_params.num_retrieved_docs
                #     )
                #     context = "\n".join(retrieved_chunks_text)
                # --- End of old logic ---

            except Exception as e:
                print(f"  Error during RAG retrieval from collection '{dynamic_collection_name}': {e}")
                context = "Error during retrieval."
                model_answer = "Error: Failed during retrieval."
                qa_error = True

            # --- LLM Question Answering ---
            if not qa_error:
                try:
                    prompt = question_prompt_template.format(context=context, question=question)
                    # Check prompt length (optional but good practice)
                    # token_count = len(question_llm_connector.tokenizer.encode(prompt)) # Assuming connector has tokenizer
                    # print(f"    Prompt token count: {token_count}")
                    model_answer = question_llm_connector.invoke(prompt)
                except Exception as e:
                    print(f"  Error during LLM QA invocation: {e}")
                    model_answer = "Error: Failed during QA generation."
                    qa_error = True

            # --- Store Intermediate Result (without evaluation yet) ---
            intermediate_entry = {
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "page": page,
                "dataset": dataset_name,
                "qa_error": qa_error # Flag if QA/Retrieval failed
            }
            dataset_intermediate_results.append(intermediate_entry)

        dataset_end_time = time.time()
        dataset_duration = dataset_end_time - dataset_start_time
        print(f"Finished QA for dataset {dataset_name} in {dataset_duration:.2f} seconds.")

        intermediate_results_by_dataset[dataset_name] = {
             "results": dataset_intermediate_results,
             "duration_qa_seconds": dataset_duration,
             "total_questions_processed": len(dataset_intermediate_results)
        }

    overall_qa_end_time = time.time()
    overall_qa_duration = overall_qa_end_time - overall_qa_start_time
    print(f"\n--- Finished Phase 1 (QA) in {overall_qa_duration:.2f} seconds ---")


    # --- Phase 2: Evaluation ---
    print("\n--- Phase 2: Evaluating all answers ---")
    overall_eval_start_time = time.time()
    evaluated_questions_count = 0
    final_results_list_for_metrics = [] # Flat list needed for calculate_metrics

    for dataset_name, dataset_data in intermediate_results_by_dataset.items():
        print(f"Evaluating Dataset: {dataset_name}")
        dataset_eval_start_time = time.time()
        evaluated_results_in_dataset = []

        for i, intermediate_result in enumerate(dataset_data["results"]):
            evaluated_questions_count += 1
            print(f"Evaluating A {evaluated_questions_count}/{total_questions_to_answer} (Dataset: {dataset_name} {i+1}/{len(dataset_data['results'])})...")

            evaluation_result = "error" # Default if skipping or error occurs
            eval_error = False

            # Only evaluate if QA didn't have an error
            if not intermediate_result.get("qa_error", False):
                try:
                    # Ensure expected_answer is a string for the evaluator
                    eval_judgment = evaluator.evaluate_answer(
                        intermediate_result["question"],
                        intermediate_result["model_answer"],
                        str(intermediate_result["expected_answer"])
                    )
                    evaluation_result = eval_judgment.strip().lower() if isinstance(eval_judgment, str) else "error"
                    print(f"  Evaluator judgment: {evaluation_result}")

                except Exception as e:
                    print(f"  Error during evaluation: {e}")
                    evaluation_result = "error"
                    eval_error = True
            else:
                print("  Skipping evaluation due to QA/Retrieval error.")
                evaluation_result = "skipped_due_to_qa_error" # More specific status
                eval_error = True # Treat as error for metrics calculation consistency if needed

            # Update the result entry with the evaluation
            final_entry = intermediate_result.copy() # Start with intermediate data
            final_entry["self_evaluation"] = evaluation_result # Add the judgment
            final_entry["eval_error"] = eval_error # Flag evaluation errors

            # Remove temporary flags if desired in final output
            # final_entry.pop("qa_error", None)
            # final_entry.pop("eval_error", None)

            evaluated_results_in_dataset.append(final_entry)
            # Add to the flat list ONLY IF NOT SKIPPED/ERROR? Or always add and let metrics handle it?
            # Let's add all entries to the flat list, calculate_metrics should handle non "yes"/"no" judgments if needed (or we filter before calling it)
            final_results_list_for_metrics.append(final_entry)


        dataset_eval_end_time = time.time()
        dataset_eval_duration = dataset_eval_end_time - dataset_eval_start_time
        # Update the dictionary for this dataset with evaluation info
        intermediate_results_by_dataset[dataset_name]["results"] = evaluated_results_in_dataset # Replace intermediate with final
        intermediate_results_by_dataset[dataset_name]["duration_eval_seconds"] = dataset_eval_duration
        print(f"Finished evaluation for dataset {dataset_name} in {dataset_eval_duration:.2f} seconds.")


    overall_eval_end_time = time.time()
    overall_eval_duration = overall_eval_end_time - overall_eval_start_time
    print(f"\n--- Finished Phase 2 (Evaluation) in {overall_eval_duration:.2f} seconds ---")

    # --- Phase 3: Calculate Overall Metrics ---
    print("\n--- Phase 3: Calculating Overall Metrics ---")
    if not final_results_list_for_metrics:
        print("No results available for metrics calculation.")
        overall_metrics = {}
    else:
        # Filter out results that couldn't be properly evaluated if necessary
        # The current calculate_metrics handles unexpected judgments by printing warnings,
        # so filtering might not be strictly needed unless you want to exclude errors entirely.
        valid_results_for_metrics = [
            r for r in final_results_list_for_metrics
            if isinstance(r.get("self_evaluation"), str) and r["self_evaluation"] in ["yes", "no"]
        ]
        if len(valid_results_for_metrics) != len(final_results_list_for_metrics):
             print(f"Warning: Calculating metrics on {len(valid_results_for_metrics)} results with valid 'yes'/'no' evaluations "
                   f"(out of {len(final_results_list_for_metrics)} total processed entries).")

        # Use the potentially filtered list for calculation
        overall_metrics = calculate_metrics(valid_results_for_metrics) # Use the flat list

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

    # --- Save Results to JSON (Include collection name for clarity) ---
    output_dir = config_loader.get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Add parameter details to filename for easier identification
    results_filename = os.path.join(output_dir, f"rag_test_results_cs{chunk_size}_os{overlap_size}_{timestamp}.json")

    overall_duration = overall_qa_duration + overall_eval_duration # Total time

    final_results_to_save = {
        "test_timestamp": timestamp,
        "question_model": question_model_name,
        "evaluator_model": evaluator_model_name,
        "rag_parameters": config_loader.get_rag_parameters(), # Save the full params used
        "chroma_collection_used": dynamic_collection_name, # Record which collection was tested
        "overall_metrics": overall_metrics,
        "overall_duration_seconds": overall_duration,
        "duration_qa_phase_seconds": overall_qa_duration,
        "duration_eval_phase_seconds": overall_eval_duration,
        "per_dataset_details": intermediate_results_by_dataset
    }

    save_results(results_filename, final_results_to_save)
    print(f"\nResults saved to: {results_filename}")


if __name__ == "__main__":
    # Ensure rag_pipeline.py has run at least once directly to perform embedding
    # Or add logic here to check if embedding needs to be run.
    print("Starting RAG Tester...")
    print("IMPORTANT: This script will attempt to load a ChromaDB collection specific to the")
    print("           'chunk_size' and 'overlap_size' currently set in the config file.")
    print("           Ensure 'rag_pipeline.py' has been run with these parameters first.")
    run_rag_test()
    print("RAG Tester finished.")