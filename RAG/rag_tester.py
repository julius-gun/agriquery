# RAG/rag_tester.py
import json
import os
import time
import hashlib
import re
import chromadb
import logging # Import logging
from typing import List, Dict, Any, Optional, Tuple

from analysis.analysis_tools import analyze_dataset_across_types, load_dataset
from evaluation.metrics import calculate_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from llm_connectors.base_llm_connector import BaseLLMConnector # Import base type for hinting
from rag_pipeline import initialize_retriever # Keep for retriever initialization
# Import specific retriever types for type checking and base type
from retrieval_pipelines.base_retriever import BaseRetriever # Assuming a base type exists
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
from retrieval_pipelines.hybrid_retriever import HybridRetriever # Import HybridRetriever
from utils.config_loader import ConfigLoader
from utils.result_manager import ResultManager

# --- Configure Logging ---
# Basic configuration, adjust level and format as needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_LLM_TYPE = "ollama" # Define default LLM type
SAVE_PROMPT_FREQUENCY = 100 # Save every Nth prompt
CHROMA_PERSIST_DIR = "chroma_db" # Define persist directory path

class RagTester:
    """
    Orchestrates the RAG testing process by iterating through configured
    parameters (models, algorithms, languages, chunk sizes, overlaps)
    and evaluating the performance. Uses logging for output.
    Refactored loop order for efficiency: model -> chunk -> overlap -> algo -> lang.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the RagTester by loading configuration and shared components.
        """
        logging.info("--- Initializing RagTester ---")
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

        self._validate_config() # Validation uses logging now

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

        logging.info("--- RagTester Initialization Complete ---")

    def _validate_config(self):
        """Checks if essential configuration parameters are present."""
        if not self.question_models_to_test:
            logging.error("Error: No question models specified in 'question_models_to_test' in config.")
            raise ValueError("Error: No question models specified in 'question_models_to_test' in config.")
        if not self.retrieval_algorithms_to_test:
            logging.error("Error: No retrieval algorithms specified in 'retrieval_algorithms_to_test' in config.")
            raise ValueError("Error: No retrieval algorithms specified in 'retrieval_algorithms_to_test' in config.")
        if not self.language_configs:
            logging.error("Error: No 'language_configs' found in config.json.")
            raise ValueError("Error: No 'language_configs' found in config.json.")
        if not self.chunk_sizes_to_test:
             logging.warning("Warning: No 'chunk_sizes_to_test' found in rag_parameters. Chunk size iteration will be skipped.")
        if not self.overlap_sizes_to_test:
             logging.warning("Warning: No 'overlap_sizes_to_test' found in rag_parameters. Overlap size iteration will be skipped.")
        if not self.config_loader.get_evaluator_model_name():
             logging.error("Error: 'evaluator_model_name' not found in config.")
             raise ValueError("Error: 'evaluator_model_name' not found in config.")
        # Add more checks as needed

    def _initialize_chromadb_client(self) -> chromadb.ClientAPI:
        """Initializes and returns the ChromaDB client."""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            logging.info(f"ChromaDB client initialized. Persistence directory: '{CHROMA_PERSIST_DIR}'")
            return client
        except Exception as e:
            logging.critical(f"FATAL: Error initializing ChromaDB client: {e}. Exiting.", exc_info=True)
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
            logging.info(f"Evaluator initialized with model: {evaluator_model_name}")
            return evaluator
        except Exception as e:
            logging.critical(f"FATAL: Error initializing evaluator with model {evaluator_model_name}: {e}. Exiting.", exc_info=True)
            raise # Re-raise after logging

    def _load_datasets(self) -> Tuple[Dict[str, List[Dict]], int]:
        """Loads the question datasets specified in the config, using logging."""
        logging.info("\n--- Loading English Question Datasets ---")
        dataset_paths = self.config_loader.get_question_dataset_paths()
        loaded_datasets = {}
        total_questions = 0
        if not dataset_paths:
             logging.error("Error: No 'question_dataset_paths' found in config.")
             raise ValueError("Error: No 'question_dataset_paths' found in config.")

        for dataset_name, dataset_path in dataset_paths.items():
            dataset = load_dataset(dataset_path)
            if dataset:
                loaded_datasets[dataset_name] = dataset
                total_questions += len(dataset)
                logging.info(f"  Loaded dataset '{dataset_name}' with {len(dataset)} questions.")
            else:
                logging.warning(f"  Warning: Failed to load dataset '{dataset_name}' from {dataset_path}.")

        if not loaded_datasets:
            logging.error("Error: No question datasets loaded. Exiting.")
            raise ValueError("Error: No question datasets loaded. Exiting.")

        logging.info(f"Total English questions to process per combination: {total_questions}")
        analyze_dataset_across_types(dataset_paths) # Analyze counts across datasets (assuming this func logs or prints internally)
        return loaded_datasets, total_questions

    def _load_prompt(self, prompt_key: str) -> str:
        """Loads a specific prompt template, using logging."""
        try:
            template = self.config_loader.load_prompt_template(prompt_key)
            logging.info(f"Loaded prompt template: '{prompt_key}'")
            return template
        except Exception as e:
            logging.critical(f"FATAL: Error loading prompt template '{prompt_key}': {e}. Exiting.", exc_info=True)
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
            # logging.debug(f"      Saved input prompt #{count} to {filepath}") # Optional debug verbosity

        except Exception as e:
            logging.error(f"      Error saving input prompt #{count}: {e}", exc_info=True)

    def _get_chroma_collection(self, base_collection_name: str, chunk_size: int, overlap_size: int) -> Optional[chromadb.Collection]:
        """Gets the specific ChromaDB collection for the given parameters."""
        dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
        logging.info(f"Attempting to get ChromaDB collection: '{dynamic_collection_name}'")
        try:
            # Note: Keyword retrieval might not strictly need a Chroma collection for querying,
            # but we need it here to *fetch the documents* for indexing.
            # Hybrid retrieval also needs it for indexing and embedding search.
            collection = self.chroma_client.get_collection(name=dynamic_collection_name)
            logging.info(f"Successfully connected to collection '{dynamic_collection_name}'.")
            return collection
        except Exception as e:
            # This error is critical if the collection is needed (embedding query or keyword indexing)
            logging.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error(f"!!! ERROR: Failed to get ChromaDB collection '{dynamic_collection_name}'.")
            logging.error(f"!!! This collection is required for the current test combination (embedding, keyword, or hybrid).")
            logging.error(f"!!! Ensure 'create_databases.py' (or rag_pipeline.py) was run with chunk={chunk_size}, overlap={overlap_size} for base '{base_collection_name}'.")
            logging.error(f"!!! Original error: {e}")
            logging.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return None # Indicate failure

    def _run_qa_phase(
        self,
        retriever: BaseRetriever, # Use BaseRetriever
        collection: Optional[chromadb.Collection], # Collection might be None if not needed for direct query *here*
        question_llm_connector: BaseLLMConnector, # Use BaseLLMConnector
        current_retrieval_algorithm: str, # Keep algorithm name for logic branching
        current_question_model_name: str, # Keep model name for logging/saving
        language: str,
        chunk_size: int, # Needed for prompt saving filename
        overlap_size: int # Needed for prompt saving filename
    ) -> Tuple[Dict[str, Dict[str, Any]], float, int]:
        """Runs the Question Answering phase for a single combination, using pre-initialized components."""
        intermediate_results_by_dataset = {}
        overall_qa_start_time = time.time()
        answered_questions_count = 0

        logging.info(f"\n--- Phase 1: Answering {self.total_questions_to_answer} English questions ---")

        for dataset_name, dataset in self.loaded_datasets.items():
            logging.info(f"\n  Processing Dataset for QA: {dataset_name}")
            dataset_intermediate_results = []
            dataset_start_time = time.time()

            for i, question_data in enumerate(dataset):
                answered_questions_count += 1
                question = question_data.get("question")
                expected_answer = question_data.get("answer")
                page = question_data.get("page", "N/A")

                if not question or expected_answer is None:
                    logging.warning(f"Warning: Skipping entry {i} in {dataset_name} due to missing question or answer.")
                    dataset_intermediate_results.append({
                        "question": question or "Missing Question", "expected_answer": expected_answer,
                        "model_answer": "Error: Missing essential data", "page": page,
                        "dataset": dataset_name, "qa_error": True
                    })
                    continue

                logging.info(f"  Answering Q {answered_questions_count}/{self.total_questions_to_answer} ({dataset_name} {i+1}/{len(dataset)}): {question[:80]}...")

                context = ""
                model_answer = ""
                qa_error = False
                prompt = ""
                retrieved_chunks_text = [] # Initialize list for retrieved docs

                # --- RAG Retrieval ---
                try:
                    # Polymorphic call to vectorize the query text
                    # Output type depends on the retriever (List[List[float]], List[str], Dict)
                    query_representation = retriever.vectorize_text(question)

                    # --- Branch based on algorithm for retrieval ---
                    if current_retrieval_algorithm == "embedding":
                        if not isinstance(retriever, EmbeddingRetriever):
                            # Log error and raise for clarity, although type hint helps
                            logging.error("Type mismatch: Retriever is not an EmbeddingRetriever for embedding algorithm.")
                            raise TypeError("Retriever is not an EmbeddingRetriever for embedding algorithm.")
                        if collection is None:
                             # This check might be redundant if collection fetch failure already skipped the combo
                             logging.error(f"ChromaDB collection is required for embedding retrieval but was not found/loaded.")
                             raise ValueError(f"ChromaDB collection is required for embedding retrieval but was not found/loaded.")

                        # EmbeddingRetriever.vectorize_text returns List[List[float]]
                        if not isinstance(query_representation, list) or not isinstance(query_representation[0], list):
                             logging.error(f"Unexpected query representation format for embedding: {type(query_representation)}")
                             raise TypeError("Unexpected query representation format for embedding.")

                        # Perform retrieval using ChromaDB directly
                        query_results = collection.query(
                            query_embeddings=query_representation, # Pass the embedding List[List[float]]
                            n_results=self.num_retrieved_docs,
                            include=['documents'] # Only need documents for context
                        )
                        if query_results and query_results.get('documents') and isinstance(query_results['documents'], list) and len(query_results['documents']) > 0:
                            retrieved_chunks_text = query_results['documents'][0] # query_results['documents'] is List[List[str]]
                            if not retrieved_chunks_text: logging.warning("      Warning: Embedding retrieval returned empty documents.")
                        else:
                            logging.warning(f"      Warning: Embedding retrieval failed or returned no documents from collection '{collection.name}'. Results: {query_results}")
                            # context = "Error: Could not retrieve context from database via embedding." # Set context later

                    elif current_retrieval_algorithm == "keyword":
                        # Check the type of the retriever instance
                        if not isinstance(retriever, KeywordRetriever):
                            logging.error("Type mismatch: Passed retriever is not a KeywordRetriever for keyword algorithm.")
                            raise TypeError("Retriever is not a KeywordRetriever for keyword algorithm.")
                        if not isinstance(query_representation, list): # KeywordRetriever returns List[str]
                             logging.error(f"Unexpected query representation format for keyword: {type(query_representation)}")
                             raise TypeError("Unexpected query representation format for keyword.")

                        # KeywordRetriever should have been indexed in _process_single_combination
                        # Retrieve relevant chunks using the internal BM25 index
                        retrieved_chunks_text, scores = retriever.retrieve_relevant_chunks(
                            query_representation=query_representation, # Pass tokenized query
                            top_k=self.num_retrieved_docs
                            # document_chunks_text is not needed if index was built with internal corpus
                        )

                        if retrieved_chunks_text:
                            logging.debug(f"      Keyword retrieval found {len(retrieved_chunks_text)} chunks with scores: {scores}") # Optional debug
                        else:
                            logging.warning(f"      Warning: Keyword retrieval returned no documents for query: '{question[:50]}...'")
                            # context = "No relevant context found via keyword search." # Set context later

                    elif current_retrieval_algorithm == "hybrid":
                        if not isinstance(retriever, HybridRetriever):
                            logging.error("Type mismatch: Passed retriever is not a HybridRetriever for hybrid algorithm.")
                            raise TypeError("Retriever is not a HybridRetriever for hybrid algorithm.")
                        if not isinstance(query_representation, dict): # HybridRetriever returns Dict
                             logging.error(f"Unexpected query representation format for hybrid: {type(query_representation)}")
                             raise TypeError("Unexpected query representation format for hybrid.")

                        # HybridRetriever should have had its keyword index built in _process_single_combination
                        # and its Chroma collection set during initialization.
                        # Retrieve relevant chunks using the internal combined logic (RRF)
                        retrieved_chunks_text, scores = retriever.retrieve_relevant_chunks(
                            query_representation=query_representation, # Pass dict with embedding and tokens
                            top_k=self.num_retrieved_docs
                            # document_chunks_text is not needed if index was built with internal corpus
                        )
                        if retrieved_chunks_text:
                            logging.debug(f"      Hybrid retrieval found {len(retrieved_chunks_text)} chunks with RRF scores: {scores}") # Optional debug
                        else:
                            logging.warning(f"      Warning: Hybrid retrieval returned no documents for query: '{question[:50]}...'")
                            # context = "No relevant context found via hybrid search." # Set context later

                    else:
                        logging.error(f"      Error: Unsupported retrieval algorithm '{current_retrieval_algorithm}' during QA.")
                        context = f"Error: Unsupported retrieval algorithm {current_retrieval_algorithm}."
                        qa_error = True

                    # --- Build Context String ---
                    if retrieved_chunks_text:
                        context = "\n".join(retrieved_chunks_text)
                    elif not qa_error: # If no error but no results
                         context = f"No relevant context found via {current_retrieval_algorithm} search."
                         logging.warning(f"      Context is empty for algorithm '{current_retrieval_algorithm}'.")
                    else: # If there was a qa_error during retrieval
                         context = f"Error during retrieval process for algorithm '{current_retrieval_algorithm}'."


                except Exception as e:
                    logging.error(f"      Error during RAG retrieval phase ({current_retrieval_algorithm}): {e}", exc_info=True)
                    context = "Error during retrieval execution."
                    model_answer = "Error: Failed during retrieval execution."
                    qa_error = True

                # --- LLM Question Answering ---
                if not qa_error:
                    llm_input_context = context # Use the context built above
                    try:
                        prompt = self.question_prompt_template.format(context=llm_input_context, question=question)

                        if answered_questions_count % SAVE_PROMPT_FREQUENCY == 0:
                            self._save_llm_input_prompt(
                                prompt_text=prompt, count=answered_questions_count, language=language,
                                model_name=current_question_model_name, algorithm=current_retrieval_algorithm,
                                chunk_size=chunk_size, overlap_size=overlap_size
                            )

                        # Use the passed question_llm_connector instance
                        model_answer = question_llm_connector.invoke(prompt)

                    except Exception as e:
                        logging.error(f"      Error during LLM QA invocation ({current_question_model_name}): {e}", exc_info=True)
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
            logging.info(f"  Finished QA for dataset {dataset_name} in {dataset_duration:.2f} seconds.")
            intermediate_results_by_dataset[dataset_name] = {
                "results": dataset_intermediate_results,
                "duration_qa_seconds": dataset_duration,
                "total_questions_processed": len(dataset_intermediate_results)
            }

        overall_qa_end_time = time.time()
        overall_qa_duration = overall_qa_end_time - overall_qa_start_time
        logging.info(f"--- Finished Phase 1 (QA) in {overall_qa_duration:.2f} seconds ---")
        return intermediate_results_by_dataset, overall_qa_duration, answered_questions_count


    def _run_evaluation_phase(
        self,
        intermediate_results_by_dataset: Dict[str, Dict[str, Any]],
        total_questions_processed_in_qa: int # Use count from QA phase
    ) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Runs the Evaluation phase for the results of a single combination."""
        if not self.evaluator:
             logging.warning("Skipping evaluation phase: Evaluator not initialized.")
             return intermediate_results_by_dataset, 0.0, []
        # Add a check to ensure the evaluator has its template
        if not self.evaluator.evaluation_prompt_template:
             logging.critical("FATAL: Skipping evaluation phase: Evaluator prompt template is missing.")
             # Return intermediate results as they are, with 0 duration and empty list for metrics
             return intermediate_results_by_dataset, 0.0, []


        logging.info(f"\n--- Phase 2: Evaluating {total_questions_processed_in_qa} answers ---")
        overall_eval_start_time = time.time()
        evaluated_questions_count = 0
        final_results_list_for_metrics = [] # Flat list for metric calculation

        for dataset_name, dataset_data in intermediate_results_by_dataset.items():
            logging.info(f"  Evaluating Dataset: {dataset_name}")
            dataset_eval_start_time = time.time()
            evaluated_results_in_dataset = [] # Store results with evaluation judgment

            for i, intermediate_result in enumerate(dataset_data["results"]):
                evaluated_questions_count += 1
                logging.info(f"    Evaluating A {evaluated_questions_count}/{total_questions_processed_in_qa} ({dataset_name} {i+1}/{len(dataset_data['results'])})...")

                evaluation_result = "error"
                eval_error = False

                # Skip evaluation if QA phase reported an error for this question
                if intermediate_result.get("qa_error", False):
                    logging.info("      Skipping evaluation due to QA/Retrieval error.")
                    evaluation_result = "skipped_due_to_qa_error"
                    eval_error = True
                    # No break here, process next item
                else:
                    try:
                        eval_judgment = self.evaluator.evaluate_answer(
                            intermediate_result["question"],
                            intermediate_result["model_answer"],
                            str(intermediate_result["expected_answer"]) # Ensure expected answer is string
                        )
                        # Normalize judgment: strip whitespace, lowercase
                        evaluation_result = eval_judgment.strip().lower() if isinstance(eval_judgment, str) else "error_invalid_type"
                        logging.info(f"      Evaluator judgment: {evaluation_result}")
                        # Add stricter validation if needed (e.g., check if exactly 'yes' or 'no')
                        if evaluation_result not in ["yes", "no"]:
                             logging.warning(f"      Warning: Evaluator returned unexpected judgment: '{evaluation_result}'")
                             # Decide how to handle: treat as error, or keep as is? Let's treat as non-metric contributing for now.
                             # evaluation_result = "error_unexpected_judgment" # Option
                             eval_error = True # Mark as error for metrics calculation later

                    except Exception as e:
                        logging.error(f"      Error during evaluation call: {e}", exc_info=True)
                        evaluation_result = "error_exception"
                        eval_error = True
                        break # Break out of the question loop on evaluation error

                final_entry = intermediate_result.copy()
                final_entry["self_evaluation"] = evaluation_result
                final_entry["eval_error"] = eval_error
                evaluated_results_in_dataset.append(final_entry)

                # Add to flat list only if evaluation resulted in a valid 'yes' or 'no' for metrics
                # And only if there wasn't a QA error or an Eval error
                if evaluation_result in ["yes", "no"] and not eval_error:
                     final_results_list_for_metrics.append(final_entry)

            dataset_eval_end_time = time.time()
            dataset_eval_duration = dataset_eval_end_time - dataset_eval_start_time
            intermediate_results_by_dataset[dataset_name]["results"] = evaluated_results_in_dataset # Update with eval results
            intermediate_results_by_dataset[dataset_name]["duration_eval_seconds"] = dataset_eval_duration
            logging.info(f"    Finished evaluation for dataset {dataset_name} in {dataset_eval_duration:.2f} seconds.")

        overall_eval_end_time = time.time()
        overall_eval_duration = overall_eval_end_time - overall_eval_start_time
        logging.info(f"--- Finished Phase 2 (Evaluation) in {overall_eval_duration:.2f} seconds ---")
        return intermediate_results_by_dataset, overall_eval_duration, final_results_list_for_metrics

    def _calculate_and_print_metrics(self, final_results_list_for_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates and prints overall metrics using logging."""
        logging.info(f"\n--- Phase 3: Calculating Overall Metrics ---")
        if not final_results_list_for_metrics:
            logging.warning("  No valid results (yes/no evaluations without errors) available for metrics calculation.")
            return {}

        # We already filtered for 'yes'/'no' and no eval_error when creating final_results_list_for_metrics
        logging.info(f"  Calculating metrics based on {len(final_results_list_for_metrics)} results with valid 'yes'/'no' evaluations and no errors.")
        try:
            # Assuming calculate_metrics takes the list of dicts and returns a dict of metrics
            overall_metrics = calculate_metrics(final_results_list_for_metrics)

            logging.info(f"\n--- Overall Evaluation Analysis ---")
            for key, value in overall_metrics.items():
                metric_name = key.replace('_', ' ').title()
                if isinstance(value, float):
                    logging.info(f"  {metric_name}: {value:.4f}")
                else:
                    logging.info(f"  {metric_name}: {value}")
            return overall_metrics
        except Exception as e:
             logging.error(f"  Error calculating metrics: {e}", exc_info=True)
             return {"metrics_calculation_error": str(e)}


    def _process_single_combination(
        self,
        retrieval_algorithm: str,
        question_model_name: str,
        language_config: Dict[str, Any],
        chunk_size: int,
        overlap_size: int,
        question_llm_connector: BaseLLMConnector, # Accept initialized connector
        retriever: BaseRetriever # Accept initialized retriever (might need indexing)
    ):
        """
        Processes a single combination of test parameters using pre-initialized components.
        Handles specific setup for KeywordRetriever and HybridRetriever indexing.
        """
        language = language_config.get("language")
        base_collection_name = language_config.get("collection_base_name")
        dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}" # Define dynamic name here

        logging.info(f"\n>>> Processing Combination: "
              f"Lang={language.upper()}, Model={question_model_name}, Algo={retrieval_algorithm}, "
              f"Chunk={chunk_size}, Overlap={overlap_size}, TopK={self.num_retrieved_docs} <<<")

        # --- Check if Results Already Exist ---
        logging.info("Checking for existing results file...")
        if self.result_manager.load_previous_results(
            retrieval_algorithm=retrieval_algorithm, language=language,
            question_model_name=question_model_name, chunk_size=chunk_size,
            overlap_size=overlap_size, num_retrieved_docs=self.num_retrieved_docs
        ):
            logging.info(f"Skipping combination: Results file already exists.")
            return # Skip to the next combination

        # --- Components are already initialized and passed in ---
        # Retriever instance is passed, but Keyword/Hybrid Retrievers need indexing here.

        # --- Get Specific Chroma Collection & Index Keyword/Hybrid Retriever ---
        # Collection is needed for embedding retrieval query AND for keyword/hybrid indexing data.
        collection = self._get_chroma_collection(base_collection_name, chunk_size, overlap_size)
        if collection is None:
             logging.warning(f"Skipping combination: Failed to get required ChromaDB collection '{dynamic_collection_name}'.")
             return

        # --- Specific setup for Keyword or Hybrid Retriever Indexing ---
        # Check if the algorithm requires indexing based on ChromaDB documents
        if retrieval_algorithm == "keyword" or retrieval_algorithm == "hybrid":
            logging.info(f"{retrieval_algorithm.capitalize()} Algorithm: Fetching documents from collection '{collection.name}' for indexing...")
            try:
                # Fetch all documents from the collection
                results = collection.get(include=['documents']) # Fetch only documents
                if results and results.get('documents'):
                    document_chunks = results['documents']
                    if document_chunks:
                         logging.info(f"Building index for {len(document_chunks)} documents...")
                         # Polymorphic call to build index (either KeywordRetriever or HybridRetriever)
                         if isinstance(retriever, (KeywordRetriever, HybridRetriever)):
                              # HybridRetriever has build_keyword_index, KeywordRetriever has build_index
                              if hasattr(retriever, 'build_keyword_index'):
                                   retriever.build_keyword_index(document_chunks) # Call HybridRetriever's method
                              elif hasattr(retriever, 'build_index'):
                                   retriever.build_index(document_chunks) # Call KeywordRetriever's method
                              else:
                                   # This case should ideally not be reached due to prior checks
                                   logging.error(f"Retriever of type {type(retriever).__name__} does not have a recognized index building method.")
                                   return # Skip combination if indexing fails

                              logging.info(f"Index built successfully for collection '{collection.name}'.")
                         else:
                              # This case should not happen if initialization in run_tests is correct
                              logging.error(f"Type mismatch: Expected KeywordRetriever or HybridRetriever but got {type(retriever).__name__}. Skipping.")
                              return

                    else:
                         logging.warning(f"Collection '{collection.name}' exists but contains no documents. {retrieval_algorithm.capitalize()} retrieval will yield no results.")
                         # Build empty index
                         if isinstance(retriever, (KeywordRetriever, HybridRetriever)):
                              if hasattr(retriever, 'build_keyword_index'):
                                   retriever.build_keyword_index([])
                              elif hasattr(retriever, 'build_index'):
                                   retriever.build_index([])
                         else:
                              logging.error(f"Cannot build empty index for retriever type {type(retriever).__name__}.")
                              return
                else:
                    logging.error(f"Failed to fetch documents from collection '{collection.name}' for {retrieval_algorithm} indexing. Skipping combination.")
                    return # Cannot proceed without documents

            except Exception as e:
                logging.error(f"Error fetching documents or building index for {retrieval_algorithm.capitalize()}Retriever from collection '{collection.name}': {e}. Skipping combination.", exc_info=True)
                return # Cannot proceed

        # --- Run QA Phase ---
        # The retriever instance (now indexed if keyword/hybrid) is passed
        intermediate_results, qa_duration, qa_count = self._run_qa_phase(
            retriever=retriever, # Pass initialized (and potentially indexed) retriever
            collection=collection, # Pass collection (used directly only by embedding logic in _run_qa_phase)
            question_llm_connector=question_llm_connector, # Pass initialized connector
            current_retrieval_algorithm=retrieval_algorithm,
            current_question_model_name=question_model_name,
            language=language,
            chunk_size=chunk_size,
            overlap_size=overlap_size
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
                "chroma_collection_used": collection.name if collection else "N/A", # Use dynamic name
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
        logging.info(f"\n--- Saving Results ---")
        self.result_manager.save_results( # Assuming ResultManager handles its own logging/printing if needed
            results=final_results_to_save,
            retrieval_algorithm=retrieval_algorithm, language=language,
            question_model_name=question_model_name, chunk_size=chunk_size,
            overlap_size=overlap_size, num_retrieved_docs=self.num_retrieved_docs
        )

        logging.info(f"\n<<< Finished Combination: Lang={language.upper()}, Model={question_model_name}, Algo={retrieval_algorithm}, Chunk={chunk_size}, Overlap={overlap_size} <<<")


    def run_tests(self):
        """
        Runs the full suite of tests based on the loaded configuration,
        iterating through all combinations with optimized loop order:
        model -> chunk -> overlap -> algo -> lang.
        Initializes components at the appropriate loop level.
        Handles specific setup for KeywordRetriever and HybridRetriever.
        """
        logging.info("\n--- Starting Test Iterations ---")
        total_combinations = (
            len(self.question_models_to_test) *
            len(self.chunk_sizes_to_test) *
            len(self.overlap_sizes_to_test) *
            len(self.retrieval_algorithms_to_test) *
            len(self.language_configs)
        )
        combination_count = 0
        # TODO: Make question LLM type configurable if needed
        question_llm_type = DEFAULT_LLM_TYPE

        # --- Start Iteration Loops (New Order) ---
        # Outermost loop: Question Model
        for model_name in self.question_models_to_test:
            logging.info(f"\n{'='*20} Testing Question Model: {model_name} {'='*20}")

            # --- Initialize Question LLM Connector (once per model) ---
            current_question_llm_connector: Optional[BaseLLMConnector] = None
            try:
                current_question_llm_connector = self.llm_connector_manager.get_connector(question_llm_type, model_name)
                logging.info(f"Successfully initialized question connector for model: {model_name}")
            except Exception as e:
                logging.error(f"Error initializing question connector for model {model_name}: {e}. Skipping this model.", exc_info=True)
                continue # Skip to the next model if connector fails

            # Next loops: Chunking parameters
            for chunk_size in self.chunk_sizes_to_test:
                for overlap_size in self.overlap_sizes_to_test:
                    logging.info(f"\n{'+'*15} Testing Chunk/Overlap: CS={chunk_size}, OS={overlap_size} (Model: {model_name}) {'+'*15}")

                    # Next loop: Retrieval Algorithm
                    for algorithm in self.retrieval_algorithms_to_test:
                        logging.info(f"\n{'-'*10} Testing Retrieval Algorithm: {algorithm.upper()} (Model: {model_name}, CS={chunk_size}, OS={overlap_size}) {'-'*10}")

                        # --- Initialize Retriever (once per algorithm within chunk/overlap/model) ---
                        # Note: Keyword/Hybrid Retrievers are initialized here but indexed later in _process_single_combination
                        current_retriever: Optional[BaseRetriever] = None # Use BaseRetriever or Any
                        try:
                            # Determine the dynamic collection name needed for Hybrid initialization
                            # We need a language config to form the name, let's peek at the first one?
                            # Or maybe initialize later inside the language loop?
                            # Let's initialize here, but Hybrid will need client/collection passed.
                            # If we initialize Hybrid here, it needs a collection name, but that depends on language.
                            # --> Decision: Initialize retriever *inside* the language loop if it's hybrid.
                            # --> Alternative: Pass client/collection name later? initialize_retriever now supports this.

                            # Let's try initializing here, passing client. Collection name will be set in HybridRetriever later if needed?
                            # No, HybridRetriever __init__ expects collection name now.
                            # --> Revised Decision: Initialize non-hybrid here, initialize hybrid inside language loop.

                            if algorithm != "hybrid":
                                # Initialize embedding or keyword retriever (don't need client/collection name at init)
                                current_retriever = initialize_retriever(algorithm)
                                logging.info(f"Initialized retriever: {type(current_retriever).__name__} for algorithm '{algorithm}'")
                            else:
                                # Hybrid retriever initialization deferred to language loop below
                                logging.info(f"Deferring HybridRetriever initialization until language loop (needs collection name).")
                                pass # Placeholder, will be initialized later

                        except Exception as e:
                            logging.error(f"Error initializing non-hybrid retriever for algorithm '{algorithm}': {e}. Skipping this algorithm for current chunk/overlap/model.", exc_info=True)
                            continue # Skip to the next algorithm if non-hybrid retriever fails

                        # Innermost loop: Language (uses the collection defined by chunk/overlap)
                        for lang_config in self.language_configs:
                            language = lang_config.get("language")
                            base_collection_name = lang_config.get("collection_base_name")
                            if not language or not base_collection_name:
                                logging.warning(f"Warning: Skipping invalid language config entry: {lang_config}")
                                continue

                            # --- Initialize Hybrid Retriever (if applicable) ---
                            # This now happens *inside* the language loop because we need the collection name
                            if algorithm == "hybrid" and current_retriever is None: # Check if not already initialized (e.g., from previous lang in this algo loop)
                                try:
                                    dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
                                    current_retriever = initialize_retriever(
                                        algorithm,
                                        chroma_client=self.chroma_client,
                                        collection_name=dynamic_collection_name
                                    )
                                    logging.info(f"Initialized retriever: {type(current_retriever).__name__} for algorithm '{algorithm}' using collection '{dynamic_collection_name}'")
                                except ValueError as ve: # Catch missing client/collection name error from initialize_retriever
                                     logging.error(f"Configuration error for HybridRetriever: {ve}. Skipping hybrid for this combination.")
                                     # Break this inner language loop for hybrid if init fails? Or just skip lang? Let's skip lang.
                                     continue # Skip this language for hybrid
                                except Exception as e:
                                    logging.error(f"Error initializing HybridRetriever for collection '{dynamic_collection_name}': {e}. Skipping hybrid for this language.", exc_info=True)
                                    continue # Skip this language for hybrid

                            # Check if retriever initialization failed in any path
                            if current_retriever is None:
                                 logging.error(f"Retriever for algorithm '{algorithm}' could not be initialized. Skipping combination.")
                                 # If hybrid failed for one lang, it might work for another, so only 'continue' here.
                                 continue # Skip to next language


                            combination_count += 1
                            logging.info(f"\n--- Running Combination {combination_count}/{total_combinations} ---")

                            # Process this specific combination, passing initialized components
                            # _process_single_combination now handles Keyword/Hybrid Retriever indexing
                            self._process_single_combination(
                                retrieval_algorithm=algorithm,
                                question_model_name=model_name,
                                language_config=lang_config,
                                chunk_size=chunk_size,
                                overlap_size=overlap_size,
                                question_llm_connector=current_question_llm_connector, # Pass instance
                                retriever=current_retriever # Pass instance (will be indexed if keyword/hybrid)
                            )

                        # Reset retriever after finishing all languages for an algorithm,
                        # especially important if hybrid was initialized inside the loop.
                        current_retriever = None


        logging.info("\n--- All Test Combinations Completed ---")


def start_rag_tests(config_path: str = "config.json"):
    """
    Initializes and runs the RagTester.
    This function serves as the main entry point for external scripts like main.py.
    """
    logging.info(f"--- Starting RAG tests via start_rag_tests (config: {config_path}) ---")
    # Wrap the core logic in a try/except block to report errors clearly
    # The RagTester init and run_tests methods already have internal error handling,
    # but this catches potential issues during the setup call itself.
    try:
        tester = RagTester(config_path=config_path)
        tester.run_tests()
        logging.info(f"--- RAG tests completed successfully via start_rag_tests ---")
        # Optionally return a status or results summary if needed later
        return True
    except Exception as e:
        # Error should have been logged by RagTester's internal handling or init
        logging.critical(f"--- RAG tests failed during execution initiated by start_rag_tests ---", exc_info=True)
        # Re-raise the exception so the caller (main.py) knows about the failure
        raise e

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure rag_pipeline.py has run at least once directly to perform embedding
    # Or add logic here to check if embedding needs to be run.
    # load the languages_to_test and print them
    # config_to_test = "config_fast.json"
    config_to_test = "config.json"

    logging.info(f"--- RAG Tester Script Start (Direct Execution) ---")
    logging.info(f"Using configuration file: {config_to_test}")

    try:
        # --- Pre-computation Check (Informational) ---
        # Load config just to print info before starting the main process
        try:
            temp_config_loader = ConfigLoader(config_to_test)
            languages_to_test_main = temp_config_loader.config.get("language_configs", [])
            rag_params_main = temp_config_loader.get_rag_parameters()
            chunk_sizes_main = rag_params_main.get("chunk_sizes_to_test", "N/A")
            overlap_sizes_main = rag_params_main.get("overlap_sizes_to_test", "N/A")
            models_to_test_main = temp_config_loader.get_question_models_to_test()
            algos_to_test_main = temp_config_loader.get_retrieval_algorithms_to_test()

            logging.info(f"\nStarting RAG Tester for:")
            logging.info(f"  Languages: {[lc.get('language', 'N/A') for lc in languages_to_test_main]}")
            logging.info(f"  Question Models: {models_to_test_main}")
            logging.info(f"  Retrieval Algorithms: {algos_to_test_main}")
            logging.info(f"  Chunk Sizes: {chunk_sizes_main}")
            logging.info(f"  Overlap Sizes: {overlap_sizes_main}")
            logging.info(f"\nIMPORTANT: This script will attempt to load ChromaDB collections specific to")
            logging.info(f"           each configured language AND the chunk/overlap parameters being tested.")
            logging.info(f"           Collection name format: [base_name]_cs[chunk_size]_os[overlap_size]")
            logging.info(f"           Ensure 'create_databases.py' or 'rag_pipeline.py' has been run with")
            logging.info(f"           combinations matching the 'chunk_sizes_to_test' and 'overlap_sizes_to_test'")
            logging.info(f"           defined in '{config_to_test}' under 'rag_parameters'.")
            logging.info(f"           These collections are needed for embedding, keyword indexing, AND hybrid retrieval.") # Updated note
        except Exception as config_ex:
             logging.error(f"Error loading configuration for pre-check: {config_ex}", exc_info=True)
             # Decide if this should prevent the run or just be a warning
             raise # Re-raise to prevent running with potentially bad config info

        # --- End Pre-computation Check ---

        # --- Initialize and Run Tester ---
        # Although we could instantiate directly here, calling the function ensures
        # the same entry point logic is used whether run directly or via main.py
        start_rag_tests(config_path=config_to_test)

    except FileNotFoundError as e:
         logging.critical(f"\nFATAL ERROR: Configuration file not found.")
         logging.critical(f"  Details: {e}")
         logging.critical("Please ensure the config file exists at the specified path.")
    except ValueError as e:
         logging.critical(f"\nFATAL ERROR: Invalid or missing configuration.")
         logging.critical(f"  Details: {e}")
         logging.critical("Please check the config file content.")
    except ImportError as e:
         # Catch missing rank_bm25 here too
         if 'rank_bm25' in str(e):
              logging.critical(f"\nFATAL ERROR: Missing dependency 'rank_bm25'.")
              logging.critical(f"  Details: {e}")
              logging.critical("Please install it using: pip install rank-bm25")
         else:
              logging.critical(f"\nFATAL ERROR: Failed to import necessary modules.")
              logging.critical(f"  Details: {e}")
              logging.critical("Please ensure all dependencies are installed and the project structure is correct.")
    except Exception as e:
        # Catch any other unexpected errors during initialization or run
        import traceback
        logging.critical(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.critical(f"FATAL ERROR: An unexpected error occurred during execution!")
        logging.critical(f"Error Type: {type(e).__name__}")
        logging.critical(f"Error Message: {e}")
        logging.critical("Traceback:")
        # Log the traceback instead of printing
        logging.critical(traceback.format_exc())
        logging.critical(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    finally:
        logging.info("--- RAG Tester Script End (Direct Execution) ---")