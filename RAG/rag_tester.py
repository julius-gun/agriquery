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
# Import the retriever initialization function (adjust path if needed)
from rag_pipeline import initialize_retriever
# Import specific retriever types for type checking or direct use if needed
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
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

    # --- Load Parameters for Iteration ---
    question_models_to_test = config_loader.get_question_models_to_test()
    retrieval_algorithms_to_test = config_loader.get_retrieval_algorithms_to_test()
    language_configs = config_loader.config.get("language_configs", [])
    rag_params_dict = config_loader.get_rag_parameters() # Get the whole dict
    # Extract fixed RAG parameters needed across iterations
    chunk_size = rag_params_dict.get("chunk_size", 2000)
    overlap_size = rag_params_dict.get("overlap_size", 50)
    num_retrieved_docs = rag_params_dict.get("num_retrieved_docs", 3)

    if not question_models_to_test:
        print("Error: No question models specified in 'question_models_to_test' in config. Exiting.")
        return
    if not retrieval_algorithms_to_test:
        print("Error: No retrieval algorithms specified in 'retrieval_algorithms_to_test' in config. Exiting.")
        return
    if not language_configs:
        print("Error: No 'language_configs' found in config.json. Cannot run tests.")
        return

    # --- Initialize Components Used Across All Tests (Once) ---
    print("--- Initializing Shared Components ---")
    llm_connector_manager = LLMConnectorManager(rag_config["llm_models"])

    # Evaluator setup (assuming the same evaluator for all tests)
    evaluator_model_name = config_loader.get_evaluator_model_name()
    if not evaluator_model_name:
        print("Error: 'evaluator_model_name' not found in config. Exiting.")
        return
    # TODO: Make evaluator LLM type configurable if needed
    evaluator_llm_type = "ollama"
    try:
        evaluator_llm_connector = llm_connector_manager.get_connector(evaluator_llm_type, evaluator_model_name)
        evaluation_prompt_template = config_loader.load_prompt_template("evaluation_prompt")
        evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template)
        print(f"Evaluator initialized with model: {evaluator_model_name}")
    except Exception as e:
        print(f"Error initializing evaluator with model {evaluator_model_name}: {e}. Exiting.")
        return

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


    # --- Initialize ChromaDB Client (Once) ---
    persist_directory = "chroma_db" # Define persist directory path
    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        print(f"ChromaDB client initialized. Persistence directory: '{persist_directory}'")
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}. Exiting.")
        return

    # --- Load Prompts (Once) ---
    try:
        question_prompt_template = config_loader.load_prompt_template("question_prompt")
    except Exception as e:
        print(f"Error loading question prompt template: {e}. Exiting.")
        return

    # --- Start Iteration Loops ---
    print("\n--- Starting Test Iterations ---")
    total_combinations = len(retrieval_algorithms_to_test) * len(question_models_to_test) * len(language_configs)
    combination_count = 0

    for current_retrieval_algorithm in retrieval_algorithms_to_test:
        print(f"\n{'='*15} Testing Retrieval Algorithm: {current_retrieval_algorithm.upper()} {'='*15}")

        # --- Initialize Retriever for the current algorithm ---
        try:
            # Use the imported initialize_retriever function
            retriever = initialize_retriever(current_retrieval_algorithm)
            print(f"Initialized retriever: {type(retriever).__name__}")
        except Exception as e:
            print(f"Error initializing retriever for algorithm '{current_retrieval_algorithm}': {e}")
            print(f"Skipping tests for algorithm: {current_retrieval_algorithm}")
            continue # Skip to the next algorithm

        for current_question_model_name in question_models_to_test:
            print(f"\n{'-'*10} Testing Question Model: {current_question_model_name} (Algorithm: {current_retrieval_algorithm}) {'-'*10}")

            # --- Initialize Question LLM Connector for the current model ---
            # TODO: Make question LLM type configurable if needed
            question_llm_type = "ollama"
            try:
                question_llm_connector = llm_connector_manager.get_connector(question_llm_type, current_question_model_name)
                print(f"Initialized question connector for model: {current_question_model_name}")
            except Exception as e:
                print(f"Error initializing question connector for model {current_question_model_name}: {e}")
                print(f"Skipping tests for model: {current_question_model_name}")
                continue # Skip to the next model

            for lang_config in language_configs:
                language = lang_config.get("language")
                base_collection_name = lang_config.get("collection_base_name")

                if not language or not base_collection_name:
                    print(f"Warning: Skipping invalid language config entry: {lang_config}")
                    continue

                combination_count += 1
                print(f"\n>>> Processing Combination {combination_count}/{total_combinations}: "
                      f"Lang={language.upper()}, Model={current_question_model_name}, Algo={current_retrieval_algorithm} <<<")

                # --- Determine Dynamic Collection Name ---
                # Name depends on chunk/overlap size used during *creation*
                dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
                print(f"Attempting to use ChromaDB collection: '{dynamic_collection_name}'")

                # --- Check if Results Already Exist ---
                print("Checking for existing results file...")
                existing_results = result_manager.load_previous_results(
                    retrieval_algorithm=current_retrieval_algorithm,
                    language=language,
                    question_model_name=current_question_model_name,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    num_retrieved_docs=num_retrieved_docs
                )

                if existing_results is not None:
                    print(f"Skipping combination: Results file already exists for "
                          f"Lang={language.upper()}, Model={current_question_model_name}, Algo={current_retrieval_algorithm}")
                    continue # Skip to the next language configuration

                # --- Get Specific Collection for this Language ---
                try:
                    # Note: Keyword retrieval might not need a Chroma collection,
                    # but embedding retrieval does. Handle this if necessary.
                    # For now, assume we always try to get it for context.
                    collection = chroma_client.get_collection(name=dynamic_collection_name)
                    print(f"Successfully connected to collection '{dynamic_collection_name}'.")
                except Exception as e:
                    # This error is critical for embedding retrieval
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"!!! ERROR: Failed to get ChromaDB collection '{dynamic_collection_name}' for language '{language}'.")
                    print(f"!!! This is required for '{current_retrieval_algorithm}' retrieval.")
                    print(f"!!! Ensure 'create_databases.py' was run with chunk={chunk_size}, overlap={overlap_size}.")
                    print(f"!!! Original error: {e}")
                    print(f"!!! Skipping combination for language: {language.upper()}")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    continue # Skip to the next language

                # --- Initialize Data Structures for this Test Combination ---
                intermediate_results_by_dataset = {}
                overall_qa_start_time = time.time()
                answered_questions_count = 0

                # --- Phase 1: Question Answering ---
                print(f"\n--- Phase 1: Answering {total_questions_to_answer} English questions "
                      f"(Lang={language.upper()}, Model={current_question_model_name}, Algo={current_retrieval_algorithm}) ---")

                for dataset_name, dataset in loaded_datasets.items():
                    print(f"\n  Processing Dataset for QA: {dataset_name}")
                    dataset_intermediate_results = []
                    dataset_start_time = time.time()

                    for i, question_data in enumerate(dataset):
                        answered_questions_count += 1
                        question = question_data.get("question")
                        expected_answer = question_data.get("answer")
                        page = question_data.get("page", "N/A") # might not be necessary to include in RAG results

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

                        # --- RAG Retrieval (Algorithm Dependent) ---
                        try:
                            if current_retrieval_algorithm == "embedding":
                                if not isinstance(retriever, EmbeddingRetriever):
                                     raise TypeError("Retriever is not an EmbeddingRetriever for embedding algorithm.")
                                question_embedding = retriever.vectorize_text(question)
                                query_results = collection.query(
                                    query_embeddings=question_embedding,
                                    n_results=num_retrieved_docs,
                                    include=['documents']
                                )
                                if query_results and query_results.get('documents') and isinstance(query_results['documents'], list) and len(query_results['documents']) > 0:
                                    retrieved_chunks_text = query_results['documents'][0]
                                    context = "\n".join(retrieved_chunks_text)
                                    if not context: print("      Warning: Embedding retrieval returned empty documents.")
                                else:
                                    print(f"      Warning: Embedding retrieval failed or returned no documents from '{dynamic_collection_name}'. Results: {query_results}")
                                    context = "Error: Could not retrieve context from database via embedding."
                                    # Decide if this is a qa_error or just empty context
                                    # qa_error = True # Let's consider it an error for now

                            elif current_retrieval_algorithm == "keyword":
                                # Placeholder logic for keyword retrieval
                                # Assumes KeywordRetriever has necessary methods, even if dummy ones
                                if not isinstance(retriever, KeywordRetriever):
                                     raise TypeError("Retriever is not a KeywordRetriever for keyword algorithm.")
                                # Keyword retrieval might not use embeddings or Chroma query
                                # It might need the raw documents from the collection or another source
                                print("      Keyword retrieval logic is currently a placeholder.")
                                # Example: Get all documents (inefficient) and let retriever filter
                                # all_docs = collection.get(include=['documents'])['documents']
                                # retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(question, None, all_docs, top_k=num_retrieved_docs)
                                # For now, just set placeholder context
                                context = f"Placeholder context for keyword retrieval of question: {question}"
                                # retrieved_chunks_text = [context] # Dummy chunk list

                            else:
                                print(f"      Error: Unsupported retrieval algorithm '{current_retrieval_algorithm}' during QA.")
                                context = f"Error: Unsupported retrieval algorithm {current_retrieval_algorithm}."
                                qa_error = True

                        except Exception as e:
                            print(f"      Error during RAG retrieval ({current_retrieval_algorithm}): {e}")
                            context = "Error during retrieval."
                            model_answer = "Error: Failed during retrieval."
                            qa_error = True

                        # --- LLM Question Answering ---
                        if not qa_error and context: # Only proceed if retrieval didn't fail AND context exists
                            try:
                                prompt = question_prompt_template.format(context=context, question=question)
                                model_answer = question_llm_connector.invoke(prompt)
                            except Exception as e:
                                print(f"      Error during LLM QA invocation ({current_question_model_name}): {e}")
                                model_answer = "Error: Failed during QA generation."
                                qa_error = True
                        elif not context and not qa_error:
                             # Handle case where retrieval succeeded but found nothing
                             print("      No context retrieved, asking model without context.")
                             try:
                                 # Ask without context - format might need adjustment if template requires context
                                 # Simple approach: provide empty context or modify prompt
                                 prompt = question_prompt_template.format(context="No context available.", question=question)
                                 model_answer = question_llm_connector.invoke(prompt)
                             except Exception as e:
                                 print(f"      Error during LLM QA invocation (no context): {e}")
                                 model_answer = "Error: Failed during QA generation (no context)."
                                 qa_error = True
                        # else: qa_error was already true, model_answer set during retrieval error

                        # --- Store Intermediate Result ---
                        intermediate_entry = {
                            "question": question,
                            "expected_answer": expected_answer,
                            "model_answer": model_answer,
                            "page": page,
                            "dataset": dataset_name,
                            "qa_error": qa_error
                            # Optionally add retrieved context/chunks here for debugging
                            # "retrieved_context": context[:500] + "..." if context else None
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
                print(f"--- Finished Phase 1 (QA) for Combination {combination_count} in {overall_qa_duration:.2f} seconds ---")


                # --- Phase 2: Evaluation ---
                print(f"\n--- Phase 2: Evaluating {total_questions_to_answer} answers for Combination {combination_count} ---")
                overall_eval_start_time = time.time()
                evaluated_questions_count = 0
                final_results_list_for_metrics = []

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
                            eval_error = True # Count as eval error if QA failed

                        final_entry = intermediate_result.copy()
                        final_entry["self_evaluation"] = evaluation_result
                        final_entry["eval_error"] = eval_error
                        evaluated_results_in_dataset.append(final_entry)
                        # Add to flat list only if evaluation was attempted (not skipped due to QA error initially)
                        # and resulted in 'yes' or 'no' for valid metrics calculation
                        if evaluation_result in ["yes", "no"]:
                             final_results_list_for_metrics.append(final_entry)

                    dataset_eval_end_time = time.time()
                    dataset_eval_duration = dataset_eval_end_time - dataset_eval_start_time
                    intermediate_results_by_dataset[dataset_name]["results"] = evaluated_results_in_dataset # Store results with eval judgment
                    intermediate_results_by_dataset[dataset_name]["duration_eval_seconds"] = dataset_eval_duration
                    print(f"    Finished evaluation for dataset {dataset_name} (for {language}) in {dataset_eval_duration:.2f} seconds.")

                overall_eval_end_time = time.time()
                overall_eval_duration = overall_eval_end_time - overall_eval_start_time
                print(f"--- Finished Phase 2 (Evaluation) for Combination {combination_count} in {overall_eval_duration:.2f} seconds ---")

                # --- Phase 3: Calculate Overall Metrics for this Language ---
                print(f"\n--- Phase 3: Calculating Overall Metrics for {language.upper()} for Combination {combination_count} ---")
                if not final_results_list_for_metrics:
                    print("  No valid results available for metrics calculation (all questions might have had QA errors or invalid evaluations).")
                    overall_metrics = {}
                else:
                    valid_results_for_metrics = [
                        r for r in final_results_list_for_metrics
                        if isinstance(r.get("self_evaluation"), str) and r["self_evaluation"] in ["yes", "no"]
                    ]
                    if len(valid_results_for_metrics) != len(final_results_list_for_metrics):
                        print(f"  Warning: Calculating metrics on {len(valid_results_for_metrics)} results with valid 'yes'/'no' evaluations "
                            f"(out of {len(final_results_list_for_metrics)} total processed entries for {language}).")
                    print(f"  Calculating metrics based on {len(final_results_list_for_metrics)} results with valid 'yes'/'no' evaluations.")
                    overall_metrics = calculate_metrics(final_results_list_for_metrics) # Use the filtered list

                    print(f"\n--- Overall Evaluation Analysis (Lang={language.upper()}, Model={current_question_model_name}, Algo={current_retrieval_algorithm}) ---")
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
                        "question_model": current_question_model_name, # Use current model
                        "evaluator_model": evaluator_model_name,
                        "retrieval_algorithm": current_retrieval_algorithm, # Use current algo
                        "chunk_size": chunk_size,
                        "overlap_size": overlap_size,
                        "num_retrieved_docs": num_retrieved_docs,
                        "chroma_collection_used": dynamic_collection_name,
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
                print(f"\n--- Saving Results for {language.upper()} for Combination {combination_count} ---")
                result_manager.save_results(
                    results=final_results_to_save,
                    retrieval_algorithm=current_retrieval_algorithm, # Pass current algo
                    language=language,
                    question_model_name=current_question_model_name, # Pass current model
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    num_retrieved_docs=num_retrieved_docs
                )
                # The actual filename will be printed by result_manager.save_results

                print(f"\n<<< Finished Combination {combination_count}/{total_combinations}: "
                      f"Lang={language.upper()}, Model={current_question_model_name}, Algo={current_retrieval_algorithm} <<<")

    print("\n--- All Test Combinations Completed ---")


if __name__ == "__main__":
    # Ensure rag_pipeline.py has run at least once directly to perform embedding
    # Or add logic here to check if embedding needs to be run.
    # load the languages_to_test and print them
    config_to_test = "config_fast.json"
    config_loader_main = ConfigLoader(config_to_test) # Renamed to avoid conflict
    languages_to_test_main = config_loader_main.config.get("language_configs", [])
    rag_params_main = config_loader_main.get_rag_parameters()
    chunk_size_main = rag_params_main.get("chunk_size", "N/A")
    overlap_size_main = rag_params_main.get("overlap_size", "N/A")
    models_to_test_main = config_loader_main.get_question_models_to_test()
    algos_to_test_main = config_loader_main.get_retrieval_algorithms_to_test()


    print(f"Starting RAG Tester for:")
    print(f"  Languages: {[lc.get('language', 'N/A') for lc in languages_to_test_main]}")
    print(f"  Question Models: {models_to_test_main}")
    print(f"  Retrieval Algorithms: {algos_to_test_main}")
    print(f"IMPORTANT: This script will attempt to load ChromaDB collections specific to")
    print(f"           each configured language and the RAG parameters in config:")
    print(f"           Chunk Size = {chunk_size_main}, Overlap Size = {overlap_size_main}")
    print(f"           Ensure 'create_databases.py' has been run with these parameters first.")

    run_rag_test(config_path = config_to_test)
    print("RAG Tester finished.")