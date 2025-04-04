# RAG/rag_tester.py
import json
import os
import time
import hashlib
import chromadb # Import chromadb

from analysis.analysis_tools import analyze_dataset_across_types, load_dataset
from evaluation.metrics import calculate_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from parameter_tuning.parameters import RagParameters
# Retriever is initialized in rag_pipeline and imported conceptually,
# but rag_tester doesn't need to re-initialize it, only use its methods if needed
# (though current logic uses collection.query directly).
# We might need to import the retriever instance if its methods are directly called later.
# from rag_pipeline import retriever # Example if needed
from utils.config_loader import ConfigLoader
from utils.result_manager import ResultManager # Import ResultManager

# --- Helper Functions for Result File Handling ---


def generate_question_key(dataset_name: str, question: str) -> str:
    """Generates a unique key for a question within a dataset."""
    # Using SHA256 hash for a consistent and filesystem-friendly key
    return hashlib.sha256(f"{dataset_name}_{question}".encode()).hexdigest()

# --- Main Test Function ---

def run_rag_test(config_path="config.json"):
    """
    Runs the RAG test for each configured language.
    For each language:
    1. Connects to the language-specific ChromaDB collection based on config parameters.
    2. Answers all ENGLISH questions using the QA LLM with context from that collection.
    3. Evaluates all answers using the Evaluator LLM.
    4. Calculates overall metrics for that language manual.
    5. Saves results using ResultManager with the new filename format.
    """
    config_loader = ConfigLoader(config_path)
    rag_config = config_loader.config # Load the entire config
    output_dir = config_loader.get_output_dir()

    # --- Initialize Result Manager (Once) ---
    result_manager = ResultManager(output_dir=output_dir)

    # --- Load Models (Once) ---
    # Pass the entire llm_models dictionary to the manager
    print("--- Initializing LLM Connectors ---")

    llm_connector_manager = LLMConnectorManager(rag_config["llm_models"])
    question_model_name = config_loader.get_question_model_name()
    question_llm_type = "ollama" # TODO: Make configurable if needed
    question_llm_connector = llm_connector_manager.get_connector(question_llm_type, question_model_name)

    evaluator_model_name = config_loader.get_evaluator_model_name()
    # Determine LLM type for evaluator model (assuming ollama, adjust if needed)
    # TODO: Make LLM type configurable per model if necessary
    evaluator_llm_type = "ollama"
    evaluator_llm_connector = llm_connector_manager.get_connector(evaluator_llm_type, evaluator_model_name)
    print("LLM Connectors Initialized.")

    # --- Load Prompts (Once) ---
    question_prompt_template = config_loader.load_prompt_template("question_prompt")
    evaluation_prompt_template = config_loader.load_prompt_template("evaluation_prompt")

    # --- Initialize Evaluator (Once) ---
    evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template)

    # --- Load English Question Datasets (Once) ---
    print("\n--- Loading English Question Datasets ---")
    dataset_paths = config_loader.get_question_dataset_paths()
    loaded_datasets = {}
    total_questions_to_answer = 0
    for dataset_name, dataset_path in dataset_paths.items():
        dataset = load_dataset(dataset_path)
        if dataset:
            loaded_datasets[dataset_name] = dataset
            total_questions_to_answer += len(dataset)
            print(f"  Loaded dataset '{dataset_name}' with {len(dataset)} questions.")
        else:
            print(f"  Warning: Failed to load dataset '{dataset_name}' from {dataset_path}.")
    if not loaded_datasets:
        print("Error: No question datasets loaded. Exiting.")
        return
    print(f"Total English questions to process per language: {total_questions_to_answer}")
    analyze_dataset_across_types(dataset_paths) # Analyze counts across datasets

    # --- Get RAG Parameters (Once) ---
    rag_params_dict = config_loader.get_rag_parameters()
    rag_params = RagParameters.from_dict(rag_params_dict) # Use RagParameters class if needed, or just dict
    # Extract parameters needed for filename and logic
    retrieval_algorithm = rag_params_dict.get("retrieval_algorithm", "embedding")
    chunk_size = rag_params_dict.get("chunk_size", 2000) # Default from config or hardcoded default
    overlap_size = rag_params_dict.get("overlap_size", 50) # Default from config or hardcoded default
    num_retrieved_docs = rag_params.num_retrieved_docs # Get from RagParameters object or dict

    # --- Initialize ChromaDB Client (Once) ---
    persist_directory = "chroma_db" # Define persist directory path
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # --- Load Language Configurations ---
    language_configs = config_loader.config.get("language_configs", [])
    if not language_configs:
        print("Error: No 'language_configs' found in config.json. Cannot run tests.")
        return

    # --- Loop Through Each Language Configuration ---
    for lang_config in language_configs:
        language = lang_config.get("language")
        base_collection_name = lang_config.get("collection_base_name")

        if not language or not base_collection_name:
            print(f"Warning: Skipping invalid language config entry: {lang_config}")
            continue

        print(f"\n{'='*20} Starting Test for Language: {language.upper()} {'='*20}")

        # --- Determine Dynamic Collection Name for this language ---
        # This name depends on chunk/overlap size used during *creation* (rag_pipeline.py or create_databases.py)
        dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
        print(f"Attempting to use ChromaDB collection: '{dynamic_collection_name}'")

        # --- Get Specific Collection for this Language ---
        try:
            collection = chroma_client.get_collection(name=dynamic_collection_name)
            print(f"Successfully connected to collection '{dynamic_collection_name}'.")
        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! ERROR: Failed to get ChromaDB collection '{dynamic_collection_name}' for language '{language}'.")
            print(f"!!! Please ensure you have run 'create_databases.py' or 'rag_pipeline.py' with:")
            print(f"!!!   'chunk_size': {chunk_size}")
            print(f"!!!   'overlap_size': {overlap_size}")
            print(f"!!!   and the correct language configuration in '{config_path}'")
            print(f"!!!   to create and embed this collection.")
            print(f"!!! Original error: {e}")
            print(f"!!! Skipping test for language: {language.upper()}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue # Skip to the next language

        # --- Initialize Data Structures for this Language Test ---
        intermediate_results_by_dataset = {}
        overall_qa_start_time = time.time()
        answered_questions_count = 0 # Reset counter for each language

        # --- Phase 1: Question Answering (Using English Qs, Language Context) ---
        print(f"\n--- Phase 1: Answering {total_questions_to_answer} English questions using {language.upper()} context ---")

        # Import retriever here if needed for vectorize_text
        # This assumes the same retriever model is used for all languages
        from retrieval_pipelines.embedding_retriever import EmbeddingRetriever # Or load dynamically based on config
        retriever = EmbeddingRetriever() # Initialize it here if needed per language, or reuse a global one if appropriate

        for dataset_name, dataset in loaded_datasets.items():
            print(f"\nProcessing Dataset for QA: {dataset_name} (against {language} context)")
            dataset_intermediate_results = []
            dataset_start_time = time.time()

            for i, question_data in enumerate(dataset):
                answered_questions_count += 1
                question = question_data.get("question")
                expected_answer = question_data.get("answer")
                page = question_data.get("page", "N/A") # Page number from English dataset might be less relevant here

                if not question or expected_answer is None:
                    print(f"Warning: Skipping entry {i} in {dataset_name} due to missing question or answer.")
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

                print(f"  Answering Q {answered_questions_count}/{total_questions_to_answer} ({dataset_name} {i+1}/{len(dataset)}): {question[:80]}...")

                context = ""
                model_answer = ""
                qa_error = False

                # --- RAG Retrieval from the current language's collection ---
                try:
                    question_embedding = retriever.vectorize_text(question) # Vectorize English question

                    # Query the language-specific collection
                    query_results = collection.query(
                        query_embeddings=question_embedding,
                        n_results=num_retrieved_docs, # Use num_retrieved_docs from params
                        include=['documents']
                    )

                    if query_results and query_results.get('documents') and isinstance(query_results['documents'], list) and len(query_results['documents']) > 0:
                         retrieved_chunks_text = query_results['documents'][0]
                         context = "\n".join(retrieved_chunks_text)
                         if not context:
                              print("    Warning: Retrieval returned empty documents from collection.")
                    else:
                         print(f"    Warning: Could not retrieve documents from DB collection '{dynamic_collection_name}'. Results: {query_results}")
                         context = "Error: Could not retrieve context from database."
                         model_answer = "Error: Failed during retrieval."
                         qa_error = True

                except Exception as e:
                    print(f"    Error during RAG retrieval from collection '{dynamic_collection_name}': {e}")
                    context = "Error during retrieval."
                    model_answer = "Error: Failed during retrieval."
                    qa_error = True

                # --- LLM Question Answering ---
                if not qa_error:
                    try:
                        # Use the English question prompt, but provide the context retrieved from the foreign language manual
                        prompt = question_prompt_template.format(context=context, question=question)
                        model_answer = question_llm_connector.invoke(prompt)
                    except Exception as e:
                        print(f"    Error during LLM QA invocation: {e}")
                        model_answer = "Error: Failed during QA generation."
                        qa_error = True

                # --- Store Intermediate Result ---
                intermediate_entry = {
                    "question": question,
                    "expected_answer": expected_answer, # English expected answer
                    "model_answer": model_answer, # Answer generated from language context
                    "page": page, # Page from English dataset source
                    "dataset": dataset_name,
                    "qa_error": qa_error
                }
                dataset_intermediate_results.append(intermediate_entry)

            dataset_end_time = time.time()
            dataset_duration = dataset_end_time - dataset_start_time
            print(f"  Finished QA for dataset {dataset_name} (against {language}) in {dataset_duration:.2f} seconds.")

            intermediate_results_by_dataset[dataset_name] = {
                 "results": dataset_intermediate_results,
                 "duration_qa_seconds": dataset_duration,
                 "total_questions_processed": len(dataset_intermediate_results)
            }

        overall_qa_end_time = time.time()
        overall_qa_duration = overall_qa_end_time - overall_qa_start_time
        print(f"--- Finished Phase 1 (QA) for {language.upper()} in {overall_qa_duration:.2f} seconds ---")


        # --- Phase 2: Evaluation ---
        print(f"\n--- Phase 2: Evaluating {total_questions_to_answer} answers for {language.upper()} ---")
        overall_eval_start_time = time.time()
        evaluated_questions_count = 0 # Reset counter for each language
        final_results_list_for_metrics = [] # Flat list for metrics calculation for this language

        for dataset_name, dataset_data in intermediate_results_by_dataset.items():
            print(f"  Evaluating Dataset: {dataset_name} (for {language})")
            dataset_eval_start_time = time.time()
            evaluated_results_in_dataset = []

            for i, intermediate_result in enumerate(dataset_data["results"]):
                evaluated_questions_count += 1
                print(f"    Evaluating A {evaluated_questions_count}/{total_questions_to_answer} ({dataset_name} {i+1}/{len(dataset_data['results'])})...")

                evaluation_result = "error"
                eval_error = False

                if not intermediate_result.get("qa_error", False):
                    try:
                        # Evaluate model's answer (from language context) against English expected answer
                        eval_judgment = evaluator.evaluate_answer(
                            intermediate_result["question"],
                            intermediate_result["model_answer"],
                            str(intermediate_result["expected_answer"])
                        )
                        evaluation_result = eval_judgment.strip().lower() if isinstance(eval_judgment, str) else "error"
                        print(f"      Evaluator judgment: {evaluation_result}")

                    except Exception as e:
                        print(f"      Error during evaluation: {e}")
                        evaluation_result = "error"
                        eval_error = True
                else:
                    print("      Skipping evaluation due to QA/Retrieval error.")
                    evaluation_result = "skipped_due_to_qa_error"
                    eval_error = True

                final_entry = intermediate_result.copy()
                final_entry["self_evaluation"] = evaluation_result
                final_entry["eval_error"] = eval_error
                evaluated_results_in_dataset.append(final_entry)
                final_results_list_for_metrics.append(final_entry) # Add to flat list for this language's metrics

            dataset_eval_end_time = time.time()
            dataset_eval_duration = dataset_eval_end_time - dataset_eval_start_time
            intermediate_results_by_dataset[dataset_name]["results"] = evaluated_results_in_dataset
            intermediate_results_by_dataset[dataset_name]["duration_eval_seconds"] = dataset_eval_duration
            print(f"    Finished evaluation for dataset {dataset_name} (for {language}) in {dataset_eval_duration:.2f} seconds.")

        overall_eval_end_time = time.time()
        overall_eval_duration = overall_eval_end_time - overall_eval_start_time
        print(f"--- Finished Phase 2 (Evaluation) for {language.upper()} in {overall_eval_duration:.2f} seconds ---")

        # --- Phase 3: Calculate Overall Metrics for this Language ---
        print(f"\n--- Phase 3: Calculating Overall Metrics for {language.upper()} ---")
        if not final_results_list_for_metrics:
            print("  No results available for metrics calculation.")
            overall_metrics = {}
        else:
            valid_results_for_metrics = [
                r for r in final_results_list_for_metrics
                if isinstance(r.get("self_evaluation"), str) and r["self_evaluation"] in ["yes", "no"]
            ]
            if len(valid_results_for_metrics) != len(final_results_list_for_metrics):
                 print(f"  Warning: Calculating metrics on {len(valid_results_for_metrics)} results with valid 'yes'/'no' evaluations "
                       f"(out of {len(final_results_list_for_metrics)} total processed entries for {language}).")

            overall_metrics = calculate_metrics(valid_results_for_metrics)

            print(f"\n--- Overall Evaluation Analysis ({language.upper()} Manual) ---")
            for key, value in overall_metrics.items():
                 if isinstance(value, float):
                     print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                 else:
                     print(f"  {key.replace('_', ' ').title()}: {value}")

        # --- Prepare Final Results Dictionary ---
        overall_duration = overall_qa_duration + overall_eval_duration
        final_results_to_save = {
            "test_run_parameters": { # Group parameters for clarity
                "language_tested": language,
                "question_model": question_model_name,
                "evaluator_model": evaluator_model_name,
                "retrieval_algorithm": retrieval_algorithm,
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "num_retrieved_docs": num_retrieved_docs,
                "chroma_collection_used": dynamic_collection_name,
                "question_datasets_used": list(dataset_paths.keys()), # List names of English datasets used
            },
            "overall_metrics": overall_metrics,
            "timing": {
                "overall_duration_seconds": overall_duration,
                "duration_qa_phase_seconds": overall_qa_duration,
                "duration_eval_phase_seconds": overall_eval_duration,
            },
            "per_dataset_details": intermediate_results_by_dataset # Results broken down by English dataset type
        }

        # --- Save Results using ResultManager ---
        print(f"\n--- Saving Results for {language.upper()} ---")
        result_manager.save_results(
            results=final_results_to_save,
            retrieval_algorithm=retrieval_algorithm,
            language=language,
            question_model_name=question_model_name,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            num_retrieved_docs=num_retrieved_docs
        )
        # The actual filename will be printed by result_manager.save_results

        print(f"{'='*20} Finished Test for Language: {language.upper()} {'='*20}")

    print("\n--- All Language Tests Completed ---")


if __name__ == "__main__":
    # Ensure rag_pipeline.py has run at least once directly to perform embedding
    # Or add logic here to check if embedding needs to be run.
    # load the languages_to_test and print them
    config_loader = ConfigLoader("config.json")
    languages_to_test = config_loader.config.get("language_configs", [])
    rag_params_main = config_loader.get_rag_parameters()
    chunk_size_main = rag_params_main.get("chunk_size", "N/A")
    overlap_size_main = rag_params_main.get("overlap_size", "N/A")

    print(f"Starting RAG Tester for the following languages: {[lc.get('language', 'N/A') for lc in languages_to_test]}")
    print(f"IMPORTANT: This script will attempt to load ChromaDB collections specific to")
    print(f"           each configured language and the RAG parameters in config:")
    print(f"           Chunk Size = {chunk_size_main}, Overlap Size = {overlap_size_main}")
    print(f"           Ensure 'create_databases.py' or 'rag_pipeline.py' has been run with these parameters first.")
    run_rag_test()
    print("RAG Tester finished.")