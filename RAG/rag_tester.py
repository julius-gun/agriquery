# RAG/rag_tester.py
import json
import os
import time
import hashlib
import re
import chromadb
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

from analysis.analysis_tools import analyze_dataset_across_types, load_dataset
from evaluation.metrics import calculate_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from rag_pipeline import initialize_retriever # Keep for retriever initialization
# Import specific retriever types for type checking
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
from utils.config_loader import ConfigLoader
from utils.result_manager import ResultManager

# --- Constants ---
DEFAULT_LLM_TYPE = "ollama" # Define default LLM type
SAVE_PROMPT_FREQUENCY = 100 # Save every Nth prompt
CHROMA_PERSIST_DIR = "chroma_db" # Define persist directory path

class RagTester:
    """
    Orchestrates the RAG testing process by iterating through configured
    parameters (algorithms, models, languages, chunk sizes, overlaps)
    and evaluating the performance.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the RagTester by loading configuration and shared components.
        """
        print("--- Initializing RagTester ---")
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.output_dir = self.config_loader.get_output_dir()

        # Load parameters for iteration
        self.question_models_to_test = self.config_loader.get_question_models_to_test()
        self.retrieval_algorithms_to_test = self.config_loader.get_retrieval_algorithms_to_test()
        self.language_configs = self.config.get("language_configs", [])
        self.rag_params_dict = self.config_loader.get_rag_parameters()
        self.chunk_sizes_to_test = self.rag_params_dict.get("chunk_sizes_to_test", [])
        self.overlap_sizes_to_test = self.rag_params_dict.get("overlap_sizes_to_test", [])
        self.num_retrieved_docs = self.rag_params_dict.get("num_retrieved_docs", 3)

        self._validate_config()

        # Initialize shared components
        self.result_manager = ResultManager(output_dir=self.output_dir)
        self.llm_connector_manager = LLMConnectorManager(self.config["llm_models"])
        self.chroma_client = self._initialize_chromadb_client()
        self.evaluator = self._initialize_evaluator() # Evaluator initialized with None template here
        self.loaded_datasets, self.total_questions_to_answer = self._load_datasets()
        self.question_prompt_template = self._load_prompt("question_prompt")
        self.evaluation_prompt_template = self._load_prompt("evaluation_prompt") # Load the template string

        if self.evaluator: # Now update the evaluator instance with the loaded template
             self.evaluator.evaluation_prompt_template = self.evaluation_prompt_template # Corrected attribute name

        print("--- RagTester Initialization Complete ---")

    def _validate_config(self):
        """Checks if essential configuration parameters are present."""
        if not self.question_models_to_test:
            raise ValueError("Error: No question models specified in 'question_models_to_test' in config.")
        if not self.retrieval_algorithms_to_test:
            raise ValueError("Error: No retrieval algorithms specified in 'retrieval_algorithms_to_test' in config.")
        if not self.language_configs:
            raise ValueError("Error: No 'language_configs' found in config.json.")
        if not self.chunk_sizes_to_test:
             print("Warning: No 'chunk_sizes_to_test' found in rag_parameters. Chunk size iteration will be skipped.")
        if not self.overlap_sizes_to_test:
             print("Warning: No 'overlap_sizes_to_test' found in rag_parameters. Overlap size iteration will be skipped.")
        if not self.config_loader.get_evaluator_model_name():
             raise ValueError("Error: 'evaluator_model_name' not found in config.")
        # Add more checks as needed

    def _initialize_chromadb_client(self) -> chromadb.ClientAPI:
        """Initializes and returns the ChromaDB client."""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            print(f"ChromaDB client initialized. Persistence directory: '{CHROMA_PERSIST_DIR}'")
            return client
        except Exception as e:
            print(f"FATAL: Error initializing ChromaDB client: {e}. Exiting.")
            raise  # Re-raise after logging

    def _initialize_evaluator(self) -> Optional[Evaluator]:
        """Initializes and returns the Evaluator instance."""
        evaluator_model_name = self.config_loader.get_evaluator_model_name()
        # TODO: Make evaluator LLM type configurable if needed
        evaluator_llm_type = DEFAULT_LLM_TYPE
        try:
            evaluator_llm_connector = self.llm_connector_manager.get_connector(evaluator_llm_type, evaluator_model_name)
            # Prompt template might not be loaded yet, pass None initially
            evaluator = Evaluator(evaluator_llm_connector, None) # Pass None for template initially
            print(f"Evaluator initialized with model: {evaluator_model_name}")
            return evaluator
        except Exception as e:
            print(f"FATAL: Error initializing evaluator with model {evaluator_model_name}: {e}. Exiting.")
            raise # Re-raise after logging

    def _load_datasets(self) -> Tuple[Dict[str, List[Dict]], int]:
        """Loads the question datasets specified in the config."""
        print("\n--- Loading English Question Datasets ---")
        dataset_paths = self.config_loader.get_question_dataset_paths()
        loaded_datasets = {}
        total_questions = 0
        if not dataset_paths:
             raise ValueError("Error: No 'question_dataset_paths' found in config.")

        for dataset_name, dataset_path in dataset_paths.items():
            dataset = load_dataset(dataset_path)
            if dataset:
                loaded_datasets[dataset_name] = dataset
                total_questions += len(dataset)
                print(f"  Loaded dataset '{dataset_name}' with {len(dataset)} questions.")
            else:
                print(f"  Warning: Failed to load dataset '{dataset_name}' from {dataset_path}.")

        if not loaded_datasets:
            raise ValueError("Error: No question datasets loaded. Exiting.")

        print(f"Total English questions to process per combination: {total_questions}")
        analyze_dataset_across_types(dataset_paths) # Analyze counts across datasets
        return loaded_datasets, total_questions

    def _load_prompt(self, prompt_key: str) -> str:
        """Loads a specific prompt template."""
        try:
            template = self.config_loader.load_prompt_template(prompt_key)
            print(f"Loaded prompt template: '{prompt_key}'")
            return template
        except Exception as e:
            print(f"FATAL: Error loading prompt template '{prompt_key}': {e}. Exiting.")
            raise # Re-raise after logging

    @staticmethod
    def _sanitize_for_filename(filename_part: str) -> str:
        """Sanitizes a string component for use in a filename."""
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename_part)
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized if sanitized else "invalid_name"

    def _save_llm_input_prompt(
        self,
        prompt_text: str,
        count: int,
        language: str,
        model_name: str,
        algorithm: str,
        chunk_size: int,
        overlap_size: int
    ):
        """Saves the input prompt text to a file."""
        try:
            prompts_dir = os.path.join(self.output_dir, "input_prompts")
            os.makedirs(prompts_dir, exist_ok=True)

            sanitized_model = self._sanitize_for_filename(model_name)
            sanitized_algo = self._sanitize_for_filename(algorithm)

            # Include chunk/overlap in filename for clarity
            filename = f"prompt_{count}_{language}_{sanitized_model}_{sanitized_algo}_cs{chunk_size}_os{overlap_size}.txt"
            filepath = os.path.join(prompts_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            # print(f"      Saved input prompt #{count} to {filepath}") # Optional verbosity

        except Exception as e:
            print(f"      Error saving input prompt #{count}: {e}")

    def _get_chroma_collection(self, base_collection_name: str, chunk_size: int, overlap_size: int) -> Optional[chromadb.Collection]:
        """Gets the specific ChromaDB collection for the given parameters."""
        dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
        print(f"Attempting to use ChromaDB collection: '{dynamic_collection_name}'")
        try:
            # Note: Keyword retrieval might not strictly need a Chroma collection,
            # but embedding retrieval does. Handle this dependency if needed.
            # For now, assume we always try to get it for context retrieval.
            collection = self.chroma_client.get_collection(name=dynamic_collection_name)
            print(f"Successfully connected to collection '{dynamic_collection_name}'.")
            return collection
        except Exception as e:
            # This error is critical for embedding retrieval using ChromaDB
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! ERROR: Failed to get ChromaDB collection '{dynamic_collection_name}'.")
            print(f"!!! This collection is required for the current test combination.")
            print(f"!!! Ensure 'create_databases.py' (or rag_pipeline.py) was run with chunk={chunk_size}, overlap={overlap_size}.")
            print(f"!!! Original error: {e}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return None # Indicate failure

    def _run_qa_phase(
        self,
        retriever: Any, # Type hint could be improved (BaseRetriever)
        collection: Optional[chromadb.Collection], # Collection might be None for non-Chroma retrievers
        question_llm_connector: Any, # Type hint could be improved (BaseLLMConnector)
        current_retrieval_algorithm: str,
        current_question_model_name: str,
        language: str,
        chunk_size: int, # Needed for prompt saving filename
        overlap_size: int # Needed for prompt saving filename
    ) -> Tuple[Dict[str, Dict[str, Any]], float, int]:
        """Runs the Question Answering phase for a single combination."""
        intermediate_results_by_dataset = {}
        overall_qa_start_time = time.time()
        answered_questions_count = 0

        print(f"\n--- Phase 1: Answering {self.total_questions_to_answer} English questions ---")

        for dataset_name, dataset in self.loaded_datasets.items():
            print(f"\n  Processing Dataset for QA: {dataset_name}")
            dataset_intermediate_results = []
            dataset_start_time = time.time()

            for i, question_data in enumerate(dataset):
                answered_questions_count += 1
                question = question_data.get("question")
                expected_answer = question_data.get("answer")
                page = question_data.get("page", "N/A")

                if not question or expected_answer is None:
                    print(f"Warning: Skipping entry {i} in {dataset_name} due to missing question or answer.")
                    dataset_intermediate_results.append({
                        "question": question or "Missing Question", "expected_answer": expected_answer,
                        "model_answer": "Error: Missing essential data", "page": page,
                        "dataset": dataset_name, "qa_error": True
                    })
                    continue

                print(f"  Answering Q {answered_questions_count}/{self.total_questions_to_answer} ({dataset_name} {i+1}/{len(dataset)}): {question[:80]}...")

                context = ""
                model_answer = ""
                qa_error = False
                prompt = ""

                # --- RAG Retrieval ---
                try:
                    if current_retrieval_algorithm == "embedding":
                        if not isinstance(retriever, EmbeddingRetriever):
                            raise TypeError("Retriever is not an EmbeddingRetriever for embedding algorithm.")
                        if collection is None:
                             raise ValueError(f"ChromaDB collection is required for embedding retrieval but was not found/loaded.")

                        question_embedding = retriever.vectorize_text(question)
                        query_results = collection.query(
                            query_embeddings=question_embedding,
                            n_results=self.num_retrieved_docs,
                            include=['documents']
                        )
                        if query_results and query_results.get('documents') and isinstance(query_results['documents'], list) and len(query_results['documents']) > 0:
                            retrieved_chunks_text = query_results['documents'][0]
                            context = "\n".join(retrieved_chunks_text)
                            if not context: print("      Warning: Embedding retrieval returned empty documents.")
                        else:
                            print(f"      Warning: Embedding retrieval failed or returned no documents from collection '{collection.name}'. Results: {query_results}")
                            context = "Error: Could not retrieve context from database via embedding."
                            # qa_error = True # Decide if this is an error

                    elif current_retrieval_algorithm == "keyword":
                        # Placeholder logic for keyword retrieval
                        # Assumes KeywordRetriever has necessary methods, even if dummy ones
                        if not isinstance(retriever, KeywordRetriever):
                            raise TypeError("Retriever is not a KeywordRetriever for keyword algorithm.")
                        # Keyword retrieval might need different inputs (e.g., all docs)
                        # This placeholder assumes it can work without a collection or needs adaptation
                        print("      Keyword retrieval logic is currently a placeholder.")
                        # Example: Fetch all docs if needed (inefficient)
                        # if collection:
                        #     all_docs = collection.get(include=['documents'])['documents']
                        #     retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(question, None, all_docs, top_k=self.num_retrieved_docs)
                        #     context = "\n".join(retrieved_chunks_text)
                        # else:
                        #     context = "Error: Keyword retrieval needs a document source (e.g., collection)."
                        #     qa_error = True
                        context = f"Placeholder context for keyword retrieval of question: {question}" # Dummy context

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
                if not qa_error:
                    llm_input_context = context if context else "No context available."
                    try:
                        prompt = self.question_prompt_template.format(context=llm_input_context, question=question)

                        if answered_questions_count % SAVE_PROMPT_FREQUENCY == 0:
                            self._save_llm_input_prompt(
                                prompt_text=prompt, count=answered_questions_count, language=language,
                                model_name=current_question_model_name, algorithm=current_retrieval_algorithm,
                                chunk_size=chunk_size, overlap_size=overlap_size
                            )

                        model_answer = question_llm_connector.invoke(prompt)

                    except Exception as e:
                        print(f"      Error during LLM QA invocation ({current_question_model_name}): {e}")
                        model_answer = f"Error: Failed during QA generation. Details: {e}"
                        qa_error = True
                        break # Break out of the question loop on LLM error

                # --- Store Intermediate Result ---
                dataset_intermediate_results.append({
                    "question": question, "expected_answer": expected_answer,
                    "model_answer": model_answer, "page": page,
                    "dataset": dataset_name, "qa_error": qa_error
                    # "retrieved_context": context[:500] + "..." if context else None, # Optional debug info
                })

            dataset_end_time = time.time()
            dataset_duration = dataset_end_time - dataset_start_time
            print(f"  Finished QA for dataset {dataset_name} in {dataset_duration:.2f} seconds.")
            intermediate_results_by_dataset[dataset_name] = {
                "results": dataset_intermediate_results,
                "duration_qa_seconds": dataset_duration,
                "total_questions_processed": len(dataset_intermediate_results)
            }

        overall_qa_end_time = time.time()
        overall_qa_duration = overall_qa_end_time - overall_qa_start_time
        print(f"--- Finished Phase 1 (QA) in {overall_qa_duration:.2f} seconds ---")
        return intermediate_results_by_dataset, overall_qa_duration, answered_questions_count


    def _run_evaluation_phase(
        self,
        intermediate_results_by_dataset: Dict[str, Dict[str, Any]],
        total_questions_processed_in_qa: int # Use count from QA phase
    ) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Runs the Evaluation phase for the results of a single combination."""
        if not self.evaluator:
             print("Skipping evaluation phase: Evaluator not initialized.")
             return intermediate_results_by_dataset, 0.0, []
        # Add a check to ensure the evaluator has its template
        if not self.evaluator.evaluation_prompt_template:
             print("FATAL: Skipping evaluation phase: Evaluator prompt template is missing.")
             # Return intermediate results as they are, with 0 duration and empty list for metrics
             return intermediate_results_by_dataset, 0.0, []


        print(f"\n--- Phase 2: Evaluating {total_questions_processed_in_qa} answers ---")
        overall_eval_start_time = time.time()
        evaluated_questions_count = 0
        final_results_list_for_metrics = [] # Flat list for metric calculation

        for dataset_name, dataset_data in intermediate_results_by_dataset.items():
            print(f"  Evaluating Dataset: {dataset_name}")
            dataset_eval_start_time = time.time()
            evaluated_results_in_dataset = [] # Store results with evaluation judgment

            for i, intermediate_result in enumerate(dataset_data["results"]):
                evaluated_questions_count += 1
                print(f"    Evaluating A {evaluated_questions_count}/{total_questions_processed_in_qa} ({dataset_name} {i+1}/{len(dataset_data['results'])})...")

                evaluation_result = "error"
                eval_error = False

                if not intermediate_result.get("qa_error", False):
                    try:
                        eval_judgment = self.evaluator.evaluate_answer(
                            intermediate_result["question"],
                            intermediate_result["model_answer"],
                            str(intermediate_result["expected_answer"]) # Ensure expected answer is string
                        )
                        # Normalize judgment: strip whitespace, lowercase
                        evaluation_result = eval_judgment.strip().lower() if isinstance(eval_judgment, str) else "error_invalid_type"
                        print(f"      Evaluator judgment: {evaluation_result}")
                        # Add stricter validation if needed (e.g., check if exactly 'yes' or 'no')
                        if evaluation_result not in ["yes", "no"]:
                             print(f"      Warning: Evaluator returned unexpected judgment: '{evaluation_result}'")
                             # Decide how to handle: treat as error, or keep as is? Let's treat as non-metric contributing for now.
                             # evaluation_result = "error_unexpected_judgment" # Option
                             eval_error = True # Mark as error for metrics

                    except Exception as e:
                        print(f"      Error during evaluation call: {e}")
                        evaluation_result = "error_exception"
                        eval_error = True
                        break # Break out of the question loop on evaluation error
                else:
                    print("      Skipping evaluation due to QA/Retrieval error.")
                    evaluation_result = "skipped_due_to_qa_error"
                    eval_error = True # Count as eval error if QA failed
                    break # Break out of the question loop on evaluation error

                final_entry = intermediate_result.copy()
                final_entry["self_evaluation"] = evaluation_result
                final_entry["eval_error"] = eval_error
                evaluated_results_in_dataset.append(final_entry)

                # Add to flat list only if evaluation resulted in a valid 'yes' or 'no' for metrics
                if evaluation_result in ["yes", "no"]:
                     final_results_list_for_metrics.append(final_entry)

            dataset_eval_end_time = time.time()
            dataset_eval_duration = dataset_eval_end_time - dataset_eval_start_time
            intermediate_results_by_dataset[dataset_name]["results"] = evaluated_results_in_dataset # Update with eval results
            intermediate_results_by_dataset[dataset_name]["duration_eval_seconds"] = dataset_eval_duration
            print(f"    Finished evaluation for dataset {dataset_name} in {dataset_eval_duration:.2f} seconds.")

        overall_eval_end_time = time.time()
        overall_eval_duration = overall_eval_end_time - overall_eval_start_time
        print(f"--- Finished Phase 2 (Evaluation) in {overall_eval_duration:.2f} seconds ---")
        return intermediate_results_by_dataset, overall_eval_duration, final_results_list_for_metrics

    def _calculate_and_print_metrics(self, final_results_list_for_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates and prints overall metrics."""
        print(f"\n--- Phase 3: Calculating Overall Metrics ---")
        if not final_results_list_for_metrics:
            print("  No valid results (yes/no evaluations) available for metrics calculation.")
            return {}

        # We already filtered for 'yes'/'no' when creating final_results_list_for_metrics
        print(f"  Calculating metrics based on {len(final_results_list_for_metrics)} results with valid 'yes'/'no' evaluations.")
        try:
            # Assuming calculate_metrics takes the list of dicts and returns a dict of metrics
            overall_metrics = calculate_metrics(final_results_list_for_metrics)

            print(f"\n--- Overall Evaluation Analysis ---")
            for key, value in overall_metrics.items():
                metric_name = key.replace('_', ' ').title()
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
            return overall_metrics
        except Exception as e:
             print(f"  Error calculating metrics: {e}")
             return {"metrics_calculation_error": str(e)}


    def _process_single_combination(
        self,
        retrieval_algorithm: str,
        question_model_name: str,
        language_config: Dict[str, Any],
        chunk_size: int,
        overlap_size: int
    ):
        """Processes a single combination of test parameters."""
        language = language_config.get("language")
        base_collection_name = language_config.get("collection_base_name")

        print(f"\n>>> Processing Combination: "
              f"Lang={language.upper()}, Model={question_model_name}, Algo={retrieval_algorithm}, "
              f"Chunk={chunk_size}, Overlap={overlap_size}, TopK={self.num_retrieved_docs} <<<")

        # --- Check if Results Already Exist ---
        print("Checking for existing results file...")
        if self.result_manager.load_previous_results(
            retrieval_algorithm=retrieval_algorithm, language=language,
            question_model_name=question_model_name, chunk_size=chunk_size,
            overlap_size=overlap_size, num_retrieved_docs=self.num_retrieved_docs
        ):
            print(f"Skipping combination: Results file already exists.")
            return # Skip to the next combination

        # --- Initialize Retriever for the current algorithm ---
        try:
            retriever = initialize_retriever(retrieval_algorithm)
            print(f"Initialized retriever: {type(retriever).__name__}")
        except Exception as e:
            print(f"Error initializing retriever for algorithm '{retrieval_algorithm}': {e}")
            print(f"Skipping combination.")
            return

        # --- Initialize Question LLM Connector ---
        # TODO: Make question LLM type configurable if needed
        question_llm_type = DEFAULT_LLM_TYPE
        try:
            question_llm_connector = self.llm_connector_manager.get_connector(question_llm_type, question_model_name)
            print(f"Initialized question connector for model: {question_model_name}")
        except Exception as e:
            print(f"Error initializing question connector for model {question_model_name}: {e}")
            print(f"Skipping combination.")
            return

        # --- Get Specific Chroma Collection ---
        # Collection is needed for embedding retrieval, potentially others
        collection = self._get_chroma_collection(base_collection_name, chunk_size, overlap_size)
        if collection is None and retrieval_algorithm == "embedding": # Check if essential collection failed
             print(f"Skipping combination due to missing required ChromaDB collection.")
             return

        # --- Run QA Phase ---
        intermediate_results, qa_duration, qa_count = self._run_qa_phase(
            retriever=retriever, collection=collection, question_llm_connector=question_llm_connector,
            current_retrieval_algorithm=retrieval_algorithm, current_question_model_name=question_model_name,
            language=language, chunk_size=chunk_size, overlap_size=overlap_size
        )

        # --- Run Evaluation Phase ---
        evaluated_results, eval_duration, metric_results_list = self._run_evaluation_phase(
            intermediate_results_by_dataset=intermediate_results,
            total_questions_processed_in_qa=qa_count
        )

        # --- Calculate Overall Metrics ---
        overall_metrics = self._calculate_and_print_metrics(metric_results_list)

        # --- Prepare Final Results Dictionary ---
        overall_duration = qa_duration + eval_duration
        final_results_to_save = {
            "test_run_parameters": {
                "language_tested": language,
                "question_model": question_model_name,
                "evaluator_model": self.config_loader.get_evaluator_model_name(),
                "retrieval_algorithm": retrieval_algorithm,
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "num_retrieved_docs": self.num_retrieved_docs,
                "chroma_collection_used": f"{base_collection_name}_cs{chunk_size}_os{overlap_size}",
            },
            "overall_metrics": overall_metrics,
            "timing": {
                "overall_duration_seconds": overall_duration,
                "duration_qa_phase_seconds": qa_duration,
                "duration_eval_phase_seconds": eval_duration,
            },
            "per_dataset_details": evaluated_results # Contains results with evaluation judgments
        }

        # --- Save Results ---
        print(f"\n--- Saving Results ---")
        self.result_manager.save_results(
            results=final_results_to_save,
            retrieval_algorithm=retrieval_algorithm, language=language,
            question_model_name=question_model_name, chunk_size=chunk_size,
            overlap_size=overlap_size, num_retrieved_docs=self.num_retrieved_docs
        )

        print(f"\n<<< Finished Combination: Lang={language.upper()}, Model={question_model_name}, Algo={retrieval_algorithm}, Chunk={chunk_size}, Overlap={overlap_size} <<<")


    def run_tests(self):
        """
        Runs the full suite of tests based on the loaded configuration,
        iterating through all combinations of parameters.
        """
        print("\n--- Starting Test Iterations ---")
        total_combinations = (
            len(self.retrieval_algorithms_to_test) *
            len(self.question_models_to_test) *
            len(self.language_configs) *
            len(self.chunk_sizes_to_test) *
            len(self.overlap_sizes_to_test)
        )
        combination_count = 0

        # --- Start Iteration Loops ---
        # Outermost loops: Chunking parameters (as they define the collection)
        for chunk_size in self.chunk_sizes_to_test:
            for overlap_size in self.overlap_sizes_to_test:
                print(f"\n{'='*20} Testing Chunk/Overlap: CS={chunk_size}, OS={overlap_size} {'='*20}")

                # Next loops: Algorithm and Model
                for algorithm in self.retrieval_algorithms_to_test:
                    print(f"\n{'+'*15} Testing Retrieval Algorithm: {algorithm.upper()} (CS={chunk_size}, OS={overlap_size}) {'+'*15}")

                    for model_name in self.question_models_to_test:
                        print(f"\n{'-'*10} Testing Question Model: {model_name} (Algo: {algorithm}, CS={chunk_size}, OS={overlap_size}) {'-'*10}")

                        # Innermost loop: Language (uses the collection defined by chunk/overlap)
                        for lang_config in self.language_configs:
                            language = lang_config.get("language")
                            if not language:
                                print(f"Warning: Skipping invalid language config entry: {lang_config}")
                                continue

                            combination_count += 1
                            print(f"\n--- Running Combination {combination_count}/{total_combinations} ---")

                            # Process this specific combination
                            self._process_single_combination(
                                retrieval_algorithm=algorithm,
                                question_model_name=model_name,
                                language_config=lang_config,
                                chunk_size=chunk_size,
                                overlap_size=overlap_size
                            )

        print("\n--- All Test Combinations Completed ---")

def start_rag_tests(config_path: str = "config.json"):
    """
    Initializes and runs the RagTester.
    This function serves as the main entry point for external scripts like main.py.
    """
    print(f"--- Starting RAG tests via start_rag_tests (config: {config_path}) ---")
    # Wrap the core logic in a try/except block to report errors clearly
    # The RagTester init and run_tests methods already have internal error handling,
    # but this catches potential issues during the setup call itself.
    try:
        tester = RagTester(config_path=config_path)
        tester.run_tests()
        print(f"--- RAG tests completed successfully via start_rag_tests ---")
        # Optionally return a status or results summary if needed later
        return True
    except Exception as e:
        # Error should have been logged by RagTester's internal handling or init
        print(f"--- RAG tests failed during execution initiated by start_rag_tests ---")
        # Re-raise the exception so the caller (main.py) knows about the failure
        raise e

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure rag_pipeline.py has run at least once directly to perform embedding
    # Or add logic here to check if embedding needs to be run.
    # load the languages_to_test and print them
    # config_to_test = "config_fast.json"
    config_to_test = "config.json"

    print(f"--- RAG Tester Script Start (Direct Execution) ---")
    print(f"Using configuration file: {config_to_test}")

    try:
        # --- Pre-computation Check (Informational) ---
        # Load config just to print info before starting the main process
        temp_config_loader = ConfigLoader(config_to_test)
        languages_to_test_main = temp_config_loader.config.get("language_configs", [])
        rag_params_main = temp_config_loader.get_rag_parameters()
        chunk_sizes_main = rag_params_main.get("chunk_sizes_to_test", "N/A")
        overlap_sizes_main = rag_params_main.get("overlap_sizes_to_test", "N/A")
        models_to_test_main = temp_config_loader.get_question_models_to_test()
        algos_to_test_main = temp_config_loader.get_retrieval_algorithms_to_test()

        print(f"\nStarting RAG Tester for:")
        print(f"  Languages: {[lc.get('language', 'N/A') for lc in languages_to_test_main]}")
        print(f"  Question Models: {models_to_test_main}")
        print(f"  Retrieval Algorithms: {algos_to_test_main}")
        print(f"  Chunk Sizes: {chunk_sizes_main}")
        print(f"  Overlap Sizes: {overlap_sizes_main}")
        print(f"\nIMPORTANT: This script will attempt to load ChromaDB collections specific to")
        print(f"           each configured language AND the chunk/overlap parameters being tested.")
        print(f"           Collection name format: [base_name]_cs[chunk_size]_os[overlap_size]")
        print(f"           Ensure 'create_databases.py' or 'rag_pipeline.py' has been run with")
        print(f"           combinations matching the 'chunk_sizes_to_test' and 'overlap_sizes_to_test'")
        print(f"           defined in '{config_to_test}' under 'rag_parameters'.")
        # --- End Pre-computation Check ---

        # --- Initialize and Run Tester (using the new function for consistency) ---
        # Although we could instantiate directly here, calling the function ensures
        # the same entry point logic is used whether run directly or via main.py
        start_rag_tests(config_path=config_to_test)

        # Original direct instantiation (also works, but less consistent):
        # tester = RagTester(config_path=config_to_test)
        # tester.run_tests()
        # print("\n--- RAG Tester Script Finished Successfully (Direct Execution) ---")


    except FileNotFoundError as e:
         print(f"\nFATAL ERROR: Configuration file not found.")
         print(f"  Details: {e}")
         print("Please ensure the config file exists at the specified path.")
    except ValueError as e:
         print(f"\nFATAL ERROR: Invalid or missing configuration.")
         print(f"  Details: {e}")
         print("Please check the config file content.")
    except ImportError as e:
         print(f"\nFATAL ERROR: Failed to import necessary modules.")
         print(f"  Details: {e}")
         print("Please ensure all dependencies are installed and the project structure is correct.")
    except Exception as e:
        # Catch any other unexpected errors during initialization or run
        import traceback
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"FATAL ERROR: An unexpected error occurred during execution!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    finally:
        print("--- RAG Tester Script End (Direct Execution) ---")