# RAG/rag_tester.py
import json
import os
import time
import hashlib
import re
import chromadb
import logging  # Import logging
from typing import List, Dict, Any, Optional, Tuple

from analysis.analysis_tools import analyze_dataset_across_types, load_dataset
from evaluation.metrics import calculate_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from llm_connectors.base_llm_connector import (
    BaseLLMConnector,
)  # Import base type for hinting
from rag_pipeline import initialize_retriever  # Keep for retriever initialization

# Import specific retriever types for type checking and base type
from retrieval_pipelines.base_retriever import (
    BaseRetriever,
)  # Assuming a base type exists
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.keyword_retrieval import KeywordRetriever
from retrieval_pipelines.hybrid_retriever import (
    HybridRetriever,
)  # Import HybridRetriever
from utils.config_loader import ConfigLoader
from utils.result_manager import ResultManager
# Import the new custom embedding function
from utils.chroma_embedding_function import HuggingFaceEmbeddingFunction

# --- Configure Logging ---
# Basic configuration, adjust level and format as needed
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants ---
DEFAULT_LLM_TYPE = "ollama"  # Define default LLM type
SAVE_PROMPT_FREQUENCY = 100  # Save every Nth prompt
CHROMA_PERSIST_DIR = "chroma_db"  # Define persist directory path
MANUALS_DIRECTORY = "manuals" # Define manuals directory path
DATASET_METRIC_KEY = "dataset_self_evaluation_success"  # Define the key name


class RagTester:
    """
    Orchestrates the RAG testing process by iterating through configured
    parameters (models, algorithms, files, extensions, chunk sizes, overlaps)
    and evaluating the performance. Uses logging for output.
    """

    def __init__(self, config_path: str = "config.json", embedding_retriever: Optional[EmbeddingRetriever] = None):
        """
        Initializes the RagTester by loading configuration and shared components.
        
        Args:
            config_path (str): The path to the configuration file.
            embedding_retriever (Optional[EmbeddingRetriever]): A pre-initialized
                EmbeddingRetriever instance to be shared.
        """
        logging.info("--- Initializing RagTester ---")
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.output_dir = self.config_loader.get_output_dir()
        self.enable_reevaluation = self.config_loader.get_enable_reevaluation()
        self.embedding_model_config = self.config_loader.get_embedding_model_config()

        # Store the shared retriever instance passed from the entry point
        self.shared_embedding_retriever = embedding_retriever
        self.chroma_embedding_function = None

        if self.shared_embedding_retriever:
            logging.info("RagTester initialized with a shared Embedding Retriever.")
            # Initialize the custom ChromaDB embedding function that wraps the shared model
            self.chroma_embedding_function = HuggingFaceEmbeddingFunction(
                embedding_retriever=self.shared_embedding_retriever
            )
        else:
            # This case occurs if the script is run in a way that doesn't provide the model.
            # We raise an error because the new design requires the model to be provided at initialization.
            logging.error("FATAL: RagTester must be initialized with an EmbeddingRetriever instance.")
            raise ValueError("RagTester requires a shared_embedding_retriever to be provided during initialization.")


        # Load parameters for iteration
        self.question_models_to_test = self.config_loader.get_question_models_to_test()
        self.retrieval_algorithms_to_test = (
            self.config_loader.get_retrieval_algorithms_to_test()
        )
        self.files_to_test = self.config_loader.get_files_to_test()
        self.file_extensions_to_test = self.config_loader.get_file_extensions_to_test()
        self.rag_params_dict = self.config_loader.get_rag_parameters()
        self.chunk_sizes_to_test = self.rag_params_dict.get("chunk_sizes_to_test", [])
        self.overlap_sizes_to_test = self.rag_params_dict.get(
            "overlap_sizes_to_test", []
        )
        self.num_retrieved_docs = self.rag_params_dict.get("num_retrieved_docs", 3)

        # Load target dataset names for metric calculation
        self.target_dataset_names = list(
            self.config_loader.get_question_dataset_paths().keys()
        )
        if not self.target_dataset_names:
            logging.warning(
                "Warning: No 'question_dataset_paths' found in config. Dataset success rates cannot be calculated, and reevaluation pruning might be affected."
            )

        self._validate_config()  # Validation uses logging now

        # Initialize shared components
        self.result_manager = ResultManager(output_dir=self.output_dir)
        self.llm_connector_manager = LLMConnectorManager(self.config["llm_models"])
        self.chroma_client = self._initialize_chromadb_client()
        self.evaluator = (
            self._initialize_evaluator()
        )  # Evaluator initialized with None template here
        self.loaded_datasets, self.total_questions_to_answer = self._load_datasets()
        self.question_prompt_template = self._load_prompt("question_prompt")
        self.evaluation_prompt_template = self._load_prompt(
            "evaluation_prompt"
        )  # Load the template string

        if self.evaluator:  # Now update the evaluator instance with the loaded template
            self.evaluator.evaluation_prompt_template = (
                self.evaluation_prompt_template
            )  # Corrected attribute name
        logging.info(
            f"Reevaluation Mode Enabled: {self.enable_reevaluation}"
        )  # Log status

        logging.info(f"Reevaluation Mode Enabled: {self.enable_reevaluation}")
        logging.info(f"Target Datasets (from current config): {self.target_dataset_names}")
        logging.info("--- RagTester Initialization Complete ---")

    def _validate_config(self):
        """Checks if essential configuration parameters are present."""
        if not self.question_models_to_test:
            logging.error(
                "Error: No question models specified in 'question_models_to_test' in config."
            )
            raise ValueError(
                "Error: No question models specified in 'question_models_to_test' in config."
            )
        if not self.retrieval_algorithms_to_test:
            logging.error(
                "Error: No retrieval algorithms specified in 'retrieval_algorithms_to_test' in config."
            )
            raise ValueError(
                "Error: No retrieval algorithms specified in 'retrieval_algorithms_to_test' in config."
            )
        if not self.files_to_test:
            logging.error("Error: No 'files_to_test' found in config.json.")
            raise ValueError("Error: No 'files_to_test' found in config.json.")
        if not self.file_extensions_to_test:
            logging.error("Error: No 'file_extensions_to_test' found in config.json.")
            raise ValueError("Error: No 'file_extensions_to_test' found in config.json.")
        if not self.chunk_sizes_to_test:
            logging.warning(
                "Warning: No 'chunk_sizes_to_test' found in rag_parameters. Chunk size iteration will be skipped."
            )
        if not self.overlap_sizes_to_test:
            logging.warning(
                "Warning: No 'overlap_sizes_to_test' found in rag_parameters. Overlap size iteration will be skipped."
            )
        if not self.config_loader.get_evaluator_model_name():
            logging.error("Error: 'evaluator_model_name' not found.")
            raise ValueError("'evaluator_model_name' not found.")
        if self.enable_reevaluation and not self.target_dataset_names:
             logging.error("Error: Reevaluation is enabled but no 'question_dataset_paths' are defined in config.")
             raise ValueError("Reevaluation requires 'question_dataset_paths' in config.")


    def _initialize_chromadb_client(self) -> chromadb.ClientAPI:
        """Initializes and returns the ChromaDB client."""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            logging.info(
                f"ChromaDB client initialized. Persistence directory: '{CHROMA_PERSIST_DIR}'"
            )
            return client
        except Exception as e:
            logging.critical(
                f"FATAL: Error initializing ChromaDB client: {e}. Exiting.",
                exc_info=True,
            )
            raise  # Re-raise after logging

    def _initialize_evaluator(self) -> Optional[Evaluator]:
        """Initializes and returns the Evaluator instance."""
        evaluator_model_name = self.config_loader.get_evaluator_model_name()
        evaluator_llm_type, _ = self.config_loader.get_llm_type_and_config(evaluator_model_name)
        if not evaluator_llm_type:
             # Fallback if model not found under specific type (should not happen with validation)
             evaluator_llm_type = DEFAULT_LLM_TYPE
             logging.warning(f"Could not determine LLM type for evaluator '{evaluator_model_name}'. Assuming '{evaluator_llm_type}'.")

        try:
            evaluator_llm_connector = self.llm_connector_manager.get_connector(
                evaluator_llm_type, evaluator_model_name
            )
            evaluator = Evaluator(evaluator_llm_connector, None) # Template loaded later
            logging.info(f"Evaluator initialized with model: {evaluator_model_name} (Type: {evaluator_llm_type})")
            return evaluator
        except Exception as e:
            logging.critical(
                f"FATAL: Error initializing evaluator with model {evaluator_model_name}: {e}. Exiting.",
                exc_info=True,
            )
            raise  # Re-raise after logging

    def _load_datasets(self) -> Tuple[Dict[str, List[Dict]], int]:
        """Loads the question datasets specified *in the current config*."""
        logging.info("\n--- Loading Configured Question Datasets ---")
        dataset_paths = self.config_loader.get_question_dataset_paths()
        loaded_datasets = {}
        total_questions = 0
        if not dataset_paths:
            logging.error("Error: No 'question_dataset_paths' found in config.")
            raise ValueError("Error: No 'question_dataset_paths' found in config.")

        for dataset_name, dataset_path in dataset_paths.items():
            if dataset_name not in self.target_dataset_names: # Redundant check, but safe
                continue
            dataset = load_dataset(dataset_path)
            if dataset:
                valid_dataset = []
                for q_data in dataset:
                    if isinstance(q_data, dict) and "question" in q_data and "answer" in q_data:
                        q_data["dataset"] = dataset_name # Add dataset origin
                        valid_dataset.append(q_data)
                    else:
                        logging.warning(f"Dataset '{dataset_name}' contains invalid item (not dict or missing keys): {q_data}. Skipping.")
                loaded_datasets[dataset_name] = valid_dataset
                total_questions += len(valid_dataset)
                logging.info(f"  Loaded dataset '{dataset_name}' with {len(valid_dataset)} valid questions.")
            else:
                logging.warning(f"  Warning: Failed to load dataset '{dataset_name}' from {dataset_path}.")

        if not loaded_datasets:
            logging.error("Error: No valid question datasets loaded. Exiting.")
            raise ValueError("No valid question datasets loaded.")

        logging.info(f"Total questions from current config: {total_questions}")
        analyze_dataset_across_types(dataset_paths)
        return loaded_datasets, total_questions

    def _load_prompt(self, prompt_key: str) -> str:
        """Loads a specific prompt template, using logging."""
        try:
            template = self.config_loader.load_prompt_template(prompt_key)
            logging.info(f"Loaded prompt template: '{prompt_key}'")
            return template
        except Exception as e:
            logging.critical(
                f"FATAL: Error loading prompt template '{prompt_key}': {e}. Exiting.",
                exc_info=True,
            )
            raise  # Re-raise after logging

    @staticmethod
    def _sanitize_for_filename(filename_part: str) -> str:
        """Sanitizes a string component for use in a filename."""
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename_part)
        sanitized = re.sub(r"[\s_]+", "_", sanitized)
        sanitized = sanitized.strip("_")
        return sanitized if sanitized else "invalid_name"

    def _save_llm_input_prompt(
        self,
        prompt_text: str,
        count: int,
        file_identifier: str,
        model_name: str,
        algorithm: str,
        chunk_size: int,
        overlap_size: int,
    ):
        """Saves the input prompt text to a file."""
        try:
            prompts_dir = os.path.join(self.output_dir, "input_prompts")
            os.makedirs(prompts_dir, exist_ok=True)

            sanitized_model = self._sanitize_for_filename(model_name)
            sanitized_algo = self._sanitize_for_filename(algorithm)

            # Include chunk/overlap in filename for clarity
            filename = f"prompt_{count}_{file_identifier}_{sanitized_model}_{sanitized_algo}_cs{chunk_size}_os{overlap_size}.txt"
            filepath = os.path.join(prompts_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            # logging.debug(f"      Saved input prompt #{count} to {filepath}") # Optional debug verbosity

        except Exception as e:
            logging.error(
                f"      Error saving input prompt #{count}: {e}", exc_info=True
            )

    def _get_chroma_collection(
        self, base_collection_name: str, chunk_size: int, overlap_size: int
    ) -> Optional[chromadb.Collection]:
        """Gets the specific ChromaDB collection for the given parameters."""
        dynamic_collection_name = (
            f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
        )
        logging.info(
            f"Attempting to get ChromaDB collection: '{dynamic_collection_name}'"
        )
        try:
            # Pass the custom embedding function to ensure Chroma uses the shared model
            collection = self.chroma_client.get_collection(
                name=dynamic_collection_name,
                embedding_function=self.chroma_embedding_function
            )
            logging.info(
                f"Successfully connected to collection '{dynamic_collection_name}'."
            )
            return collection
        except Exception as e:
            # This error is critical if the collection is needed (embedding query or keyword indexing)
            logging.error(
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            logging.error(
                f"!!! ERROR: Failed to get ChromaDB collection '{dynamic_collection_name}'."
            )
            logging.error(
                f"!!! This collection is required for the current test combination (embedding, keyword, or hybrid)."
            )
            logging.error(
                f"!!! Ensure 'create_databases.py' was run with chunk={chunk_size}, overlap={overlap_size} for base '{base_collection_name}'."
            )
            logging.error(f"!!! Original error: {e}")
            logging.error(
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            return None  # Indicate failure

    def _handle_reevaluation(
        self,
        retrieval_algorithm: str,
        file_identifier: str,
        question_model_name: str,
        chunk_size: int,
        overlap_size: int,
        num_retrieved_docs: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Loads existing results, prunes questions based on current datasets,
        and identifies questions needing answers. Only prunes datasets listed
        in the current config (self.target_dataset_names).

        Returns:
            - Pruned previous results dict, or None if no file exists or structure is invalid.
            - List of question dicts from current datasets needing answers, or None if file not found/invalid.
              Returns an empty list if all current questions are answered.
        """
        logging.info("--- Reevaluation Mode: Handling existing results ---")
        previous_results = self.result_manager.load_previous_results(
            retrieval_algorithm,
            file_identifier,
            question_model_name,
            chunk_size,
            overlap_size,
            num_retrieved_docs,
        )

        if previous_results is None:
            logging.info(
                "No previous results file found. Proceeding with a normal run for this combination."
            )
            return None, None  # Indicate normal run

        if "per_dataset_details" not in previous_results:
            logging.warning(
                "Previous results file exists but lacks 'per_dataset_details'. Cannot reevaluate. Proceeding with normal run."
            )
            # Treat as if no file exists for reevaluation purposes
            return None, None

        # Get questions from *currently configured* datasets
        current_questions_sets = {
            name: {q["question"] for q in data if isinstance(q, dict) and "question" in q}
            for name, data in self.loaded_datasets.items()
        }
        logging.info(f"Current datasets in config: {self.target_dataset_names}")

        questions_to_answer_list = []
        pruned_previous_questions_map = {} # Map: dataset_name -> set of questions in pruned results

        # Prune results: Iterate through datasets *in the results file*
        datasets_in_results = list(previous_results["per_dataset_details"].keys())
        for dataset_name in datasets_in_results:
            # --- Pruning Logic ---
            if dataset_name not in self.target_dataset_names:
                # Dataset exists in results but NOT in current config. Keep it, but don't prune/check for missing Qs.
                logging.info(f"Dataset '{dataset_name}' found in results but not in current config. Preserving existing results; excluding from re-answering & metrics.")
                # Store its questions to avoid adding them if they somehow match a current dataset Q
                dataset_content = previous_results["per_dataset_details"][dataset_name]
                if "results" in dataset_content and isinstance(dataset_content["results"], list):
                     pruned_previous_questions_map[dataset_name] = {
                         res["question"] for res in dataset_content["results"]
                         if isinstance(res, dict) and "question" in res
                     }
                else:
                     pruned_previous_questions_map[dataset_name] = set()
                continue # Leave this dataset untouched in the structure

            # Dataset exists in results AND is in the current config. Prune it.
            logging.info(f"Pruning results for dataset '{dataset_name}' based on current questions...")
            dataset_content = previous_results["per_dataset_details"][dataset_name]
            if "results" not in dataset_content or not isinstance(dataset_content["results"], list):
                logging.warning(f"Dataset '{dataset_name}' in results has invalid 'results' list. Treating as empty for pruning.")
                pruned_results = []
            else:
                results_list = dataset_content["results"]
                current_q_set = current_questions_sets.get(dataset_name, set()) # Should exist
                if not current_q_set:
                    logging.warning(f"No questions loaded for '{dataset_name}' from current config, despite being targeted. Removing all previous results for it.")
                    pruned_results = []
                else:
                    original_count = len(results_list)
                    pruned_results = [
                        res for res in results_list
                        if isinstance(res, dict) and res.get("question") in current_q_set
                    ]
                    removed_count = original_count - len(pruned_results)
                    logging.info(f"  Removed {removed_count} results (questions not in current '{dataset_name}'). Kept {len(pruned_results)}.")

            # Update the results in the dictionary and store pruned questions
            previous_results["per_dataset_details"][dataset_name]["results"] = pruned_results
            pruned_previous_questions_map[dataset_name] = {
                res["question"] for res in pruned_results if isinstance(res, dict) and "question" in res
            }

        # Identify missing questions by comparing *current* datasets against *pruned* results
        logging.info("Identifying missing questions from current datasets...")
        for dataset_name, current_questions_data in self.loaded_datasets.items():
            # dataset_name is guaranteed to be in self.target_dataset_names here
            answered_questions_in_pruned = pruned_previous_questions_map.get(dataset_name, set())
            missing_count = 0
            for question_data in current_questions_data: # question_data is a dict {'question': ..., 'answer': ..., 'page': ..., 'dataset': ...}
                if isinstance(question_data, dict) and "question" in question_data:
                    if question_data["question"] not in answered_questions_in_pruned:
                        questions_to_answer_list.append(question_data) # Add the full dict
                        missing_count += 1
                else:
                    logging.warning(f"Skipping invalid item in loaded dataset '{dataset_name}': {question_data}")

            if missing_count > 0:
                logging.info(
                    f"  Found {missing_count} questions from '{dataset_name}' that need to be answered."
                )

        total_missing = len(questions_to_answer_list)
        if total_missing == 0:
            logging.info(
                "All questions from current datasets are already present in the results file."
            )
        else:
            logging.info(f"Total questions to answer/evaluate: {total_missing}")

        return previous_results, questions_to_answer_list

    def _run_qa_phase(
        self, retriever: BaseRetriever, collection: Optional[chromadb.Collection],
        question_llm_connector: BaseLLMConnector, current_retrieval_algorithm: str,
        current_question_model_name: str, file_identifier: str, chunk_size: int, overlap_size: int,
        # Parameter to accept specific questions/datasets to process
        questions_to_process: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], float, int]:
        """
        Runs the Question Answering phase.
        If 'questions_to_process' is provided, only processes those questions.
        Otherwise, processes all questions from 'self.loaded_datasets'.
        """
        intermediate_results_by_dataset = {}
        overall_qa_start_time = time.time()
        answered_questions_count = 0

        # Determine which questions to process
        if questions_to_process is not None:
            # Group the provided questions by dataset for processing structure
            datasets_to_iterate = {}
            for q_data in questions_to_process:
                ds_name = q_data.get("dataset", "unknown_dataset")
                if ds_name not in datasets_to_iterate:
                    datasets_to_iterate[ds_name] = []
                datasets_to_iterate[ds_name].append(q_data)
            total_questions_in_scope = len(questions_to_process)
            logging.info(f"\n--- Phase 1 (Reevaluation QA): Answering {total_questions_in_scope} missing questions ---")
        else:
            # Process all loaded datasets
            datasets_to_iterate = self.loaded_datasets
            total_questions_in_scope = self.total_questions_to_answer
            logging.info(f"\n--- Phase 1 (Full QA): Answering {total_questions_in_scope} questions ---")

        if not datasets_to_iterate:
            logging.info("No questions to process in QA phase.")
            return {}, 0.0, 0

        for dataset_name, dataset_questions in datasets_to_iterate.items():
            if not dataset_questions: continue # Skip empty datasets

            logging.info(f"\n  Processing QA for Dataset: {dataset_name} ({len(dataset_questions)} questions)")
            dataset_intermediate_results = []
            dataset_start_time = time.time()

            for i, question_data in enumerate(dataset_questions):
                answered_questions_count += 1
                question = question_data.get("question")
                # Expected answer might be needed later, ensure it's present
                expected_answer = question_data.get("answer")
                page = question_data.get("page", "N/A")
                # Dataset name should already be in question_data if _load_datasets was used
                q_dataset_name = question_data.get(
                    "dataset", dataset_name
                )  # Fallback to loop key

                if not question or expected_answer is None: # Check both Q & A
                    logging.warning(f"Skipping entry in {q_dataset_name} due to missing question/answer: {question_data}")
                    dataset_intermediate_results.append(
                        {
                            "question": question or "Missing Question",
                            "expected_answer": expected_answer,
                            "model_answer": "Error: Missing essential data",
                            "page": page,
                            "dataset": dataset_name,
                            "qa_error": True,
                        }
                    )
                    continue

                logging.info(f"  Answering Q {answered_questions_count}/{total_questions_in_scope} ({q_dataset_name} {i + 1}/{len(dataset_questions)}): {question[:80]}...")

                context = ""
                model_answer = ""
                qa_error = False
                prompt = ""
                retrieved_chunks_text = []  # Initialize list for retrieved docs

                # --- RAG Retrieval ---
                try:
                    # Polymorphic call to vectorize the query text
                    # Output type depends on the retriever (List[List[float]], List[str], Dict)
                    query_representation = retriever.vectorize_query(question)

                    # --- Branch based on algorithm for retrieval ---
                    if current_retrieval_algorithm == "embedding":
                        if not isinstance(retriever, EmbeddingRetriever):
                            # Log error and raise for clarity, although type hint helps
                            logging.error(
                                "Type mismatch: Retriever is not an EmbeddingRetriever for embedding algorithm."
                            )
                            raise TypeError(
                                "Retriever is not an EmbeddingRetriever for embedding algorithm."
                            )
                        if collection is None:
                            # This check might be redundant if collection fetch failure already skipped the combo
                            logging.error(
                                f"ChromaDB collection is required for embedding retrieval but was not found/loaded."
                            )
                            raise ValueError(
                                f"ChromaDB collection is required for embedding retrieval but was not found/loaded."
                            )

                        # EmbeddingRetriever.vectorize_query returns List[List[float]]
                        if not isinstance(query_representation, list) or not isinstance(
                            query_representation[0], list
                        ):
                            logging.error(
                                f"Unexpected query representation format for embedding: {type(query_representation)}"
                            )
                            raise TypeError(
                                "Unexpected query representation format for embedding."
                            )

                        # Perform retrieval using ChromaDB directly
                        query_results = collection.query(
                            query_embeddings=query_representation,  # Pass the embedding List[List[float]]
                            n_results=self.num_retrieved_docs,
                            include=["documents"],  # Only need documents for context
                        )
                        if (
                            query_results
                            and query_results.get("documents")
                            and isinstance(query_results["documents"], list)
                            and len(query_results["documents"]) > 0
                        ):
                            retrieved_chunks_text = query_results["documents"][
                                0
                            ]  # query_results['documents'] is List[List[str]]
                            if not retrieved_chunks_text:
                                logging.warning(
                                    "      Warning: Embedding retrieval returned empty documents."
                                )
                        else:
                            logging.warning(
                                f"      Warning: Embedding retrieval failed or returned no documents from collection '{collection.name}'. Results: {query_results}"
                            )
                            # context = "Error: Could not retrieve context from database via embedding." # Set context later

                    elif current_retrieval_algorithm == "keyword":
                        # Check the type of the retriever instance
                        if not isinstance(retriever, KeywordRetriever):
                            logging.error(
                                "Type mismatch: Passed retriever is not a KeywordRetriever for keyword algorithm."
                            )
                            raise TypeError(
                                "Retriever is not a KeywordRetriever for keyword algorithm."
                            )
                        if not isinstance(
                            query_representation, list
                        ):  # KeywordRetriever returns List[str]
                            logging.error(
                                f"Unexpected query representation format for keyword: {type(query_representation)}"
                            )
                            raise TypeError(
                                "Unexpected query representation format for keyword."
                            )

                        # KeywordRetriever should have been indexed in _process_single_combination
                        # Retrieve relevant chunks using the internal BM25 index
                        retrieved_chunks_text, scores = (
                            retriever.retrieve_relevant_chunks(
                                query_representation=query_representation,  # Pass tokenized query
                                top_k=self.num_retrieved_docs,
                                # document_chunks_text is not needed if index was built with internal corpus
                            )
                        )

                        if retrieved_chunks_text:
                            logging.debug(
                                f"      Keyword retrieval found {len(retrieved_chunks_text)} chunks with scores: {scores}"
                            )  # Optional debug
                        else:
                            logging.warning(
                                f"      Warning: Keyword retrieval returned no documents for query: '{question[:50]}...'"
                            )
                            # context = "No relevant context found via keyword search." # Set context later

                    elif current_retrieval_algorithm == "hybrid":
                        if not isinstance(retriever, HybridRetriever):
                            logging.error(
                                "Type mismatch: Passed retriever is not a HybridRetriever for hybrid algorithm."
                            )
                            raise TypeError(
                                "Retriever is not a HybridRetriever for hybrid algorithm."
                            )
                        if not isinstance(
                            query_representation, dict
                        ):  # HybridRetriever returns Dict
                            logging.error(
                                f"Unexpected query representation format for hybrid: {type(query_representation)}"
                            )
                            raise TypeError(
                                "Unexpected query representation format for hybrid."
                            )

                        # HybridRetriever should have had its keyword index built in _process_single_combination
                        # and its Chroma collection set during initialization.
                        # Retrieve relevant chunks using the internal combined logic (RRF)
                        # *** Modify this line to capture scores ***
                        retrieved_chunks_text, scores = ( # Changed _ to scores
                            retriever.retrieve_relevant_chunks(
                                query_representation=query_representation,  # Pass dict with embedding and tokens
                                top_k=self.num_retrieved_docs,
                                # document_chunks_text is not needed if index was built with internal corpus
                            )
                        )
                        if retrieved_chunks_text:
                            logging.debug(
                                f"      Hybrid retrieval found {len(retrieved_chunks_text)} chunks with RRF scores: {scores}"
                            )  # Optional debug
                        else:
                            logging.warning(
                                f"      Warning: Hybrid retrieval returned no documents for query: '{question[:50]}...'"
                            )
                            # context = "No relevant context found via hybrid search." # Set context later

                    else:
                        logging.error(
                            f"      Error: Unsupported retrieval algorithm '{current_retrieval_algorithm}' during QA."
                        )
                        context = f"Error: Unsupported retrieval algorithm {current_retrieval_algorithm}."
                        qa_error = True

                    # --- Build Context String ---
                    if retrieved_chunks_text:
                        context = "\n".join(retrieved_chunks_text)
                    elif not qa_error:  # If no error but no results
                        context = f"No relevant context found via {current_retrieval_algorithm} search."
                        logging.warning(
                            f"      Context is empty for algorithm '{current_retrieval_algorithm}'."
                        )
                    else:  # If there was a qa_error during retrieval
                        context = f"Error during retrieval process for algorithm '{current_retrieval_algorithm}'."

                except Exception as e:
                    logging.error(
                        f"      Error during RAG retrieval phase ({current_retrieval_algorithm}): {e}",
                        exc_info=True,
                    )
                    context = "Error during retrieval execution."
                    model_answer = "Error: Failed during retrieval execution."
                    qa_error = True

                # --- LLM Question Answering ---
                if not qa_error:
                    llm_input_context = context  # Use the context built above
                    try:
                        prompt = self.question_prompt_template.format(
                            context=llm_input_context, question=question
                        )

                        if answered_questions_count % SAVE_PROMPT_FREQUENCY == 0:
                            self._save_llm_input_prompt(
                                prompt_text=prompt,
                                count=answered_questions_count,
                                file_identifier=file_identifier,
                                model_name=current_question_model_name,
                                algorithm=current_retrieval_algorithm,
                                chunk_size=chunk_size,
                                overlap_size=overlap_size,
                            )

                        # Use the passed question_llm_connector instance
                        model_answer = question_llm_connector.invoke(prompt)

                    except Exception as e:
                        logging.error(
                            f"      Error during LLM QA invocation ({current_question_model_name}): {e}",
                            exc_info=True,
                        )
                        model_answer = (
                            f"Error: Failed during QA generation. Details: {e}"
                        )
                        qa_error = True
                        break  # Break out of the question loop on LLM error

                # --- Store Intermediate Result ---
                dataset_intermediate_results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "page": page,
                        "dataset": dataset_name,
                        "qa_error": qa_error,
                        # "retrieved_context": context[:500] + "..." if context else None, # Optional debug info
                    }
                )

            dataset_end_time = time.time()
            dataset_duration = dataset_end_time - dataset_start_time
            logging.info(
                f"  Finished QA for dataset {dataset_name} in {dataset_duration:.2f} seconds."
            )
            # Use the correct dataset name from the question data for grouping results
            # Group results by the actual dataset name from the question data
            grouping_dataset_name = dataset_questions[0]["dataset"] if dataset_questions else dataset_name
            intermediate_results_by_dataset[grouping_dataset_name] = {
                "results": dataset_intermediate_results,
                "duration_qa_seconds": dataset_duration,
                "total_questions_processed": len(dataset_intermediate_results),
            }

        overall_qa_end_time = time.time()
        overall_qa_duration = overall_qa_end_time - overall_qa_start_time
        log_phase = "Reevaluation QA" if questions_to_process is not None else "Full QA"
        logging.info(f"--- Finished Phase 1 ({log_phase}) in {overall_qa_duration:.2f} seconds ---")

        logging.info(
            f"--- Finished Phase 1 (QA) in {overall_qa_duration:.2f} seconds ---"
        )
        return (
            intermediate_results_by_dataset,
            overall_qa_duration,
            answered_questions_count,
        )

    def _run_evaluation_phase(
        self,
        intermediate_results_by_dataset: Dict[str, Dict[str, Any]],
        total_questions_processed_in_qa: int,  # Use count from QA phase
    ) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Runs the Evaluation phase for the results of a single QA run."""
        if not self.evaluator:
            logging.warning("Skipping evaluation phase: Evaluator not initialized.")
            return intermediate_results_by_dataset, 0.0, []
        # Add a check to ensure the evaluator has its template
        if not self.evaluator or not self.evaluator.evaluation_prompt_template:
            logging.warning("Skipping evaluation phase: Evaluator not initialized or template missing.")
            return intermediate_results_by_dataset, 0.0, []

        logging.info(
            f"\n--- Phase 2: Evaluating {total_questions_processed_in_qa} answers ---"
        )
        overall_eval_start_time = time.time()
        evaluated_questions_count = 0
        final_results_list_for_metrics = []  # Flat list for metric calculation

        for dataset_name, dataset_data in intermediate_results_by_dataset.items():
            total_in_dataset = len(dataset_data.get("results", []))
            logging.info(f"  Evaluating Dataset: {dataset_name} ({total_in_dataset} results)")
            dataset_eval_start_time = time.time()
            evaluated_results_in_dataset = []  # Store results with evaluation judgment

            for i, intermediate_result in enumerate(dataset_data.get("results", [])):
                evaluated_questions_count += 1
                logging.info(f"    Evaluating A {evaluated_questions_count}/{total_questions_processed_in_qa} ({dataset_name} {i + 1}/{total_in_dataset})...")

                evaluation_result = "error"
                eval_error = False

                # Skip evaluation if QA phase reported an error for this question
                if intermediate_result.get("qa_error", False):
                    logging.info("      Skipping evaluation due to prior QA/Retrieval error.")
                    evaluation_result = "skipped_due_to_qa_error"
                    eval_error = True
                    # No break here, process next item
                else:
                    try:
                        eval_judgment = self.evaluator.evaluate_answer(
                            intermediate_result["question"],
                            intermediate_result["model_answer"],
                            str(
                                intermediate_result["expected_answer"]
                            ),  # Ensure expected answer is string
                        )
                        # Normalize judgment: strip whitespace, lowercase
                        evaluation_result = (
                            eval_judgment.strip().lower()
                            if isinstance(eval_judgment, str)
                            else "error_invalid_type"
                        )
                        logging.info(f"      Evaluator judgment: {evaluation_result}")
                        # Add stricter validation if needed (e.g., check if exactly 'yes' or 'no')
                        if evaluation_result not in ["yes", "no"]:
                            logging.warning(
                                f"      Warning: Evaluator returned unexpected judgment: '{evaluation_result}'"
                            )
                            # Decide how to handle: treat as error, or keep as is? Let's treat as non-metric contributing for now.
                            # evaluation_result = "error_unexpected_judgment" # Option
                            eval_error = (
                                True  # Mark as error for metrics calculation later
                            )

                    except Exception as e:
                        logging.error(
                            f"      Error during evaluation call: {e}", exc_info=True
                        )
                        evaluation_result = "error_exception"
                        eval_error = True
                        break  # Break out of the question loop on evaluation error

                final_entry = intermediate_result.copy()
                final_entry["self_evaluation"] = evaluation_result
                final_entry["eval_error"] = eval_error
                evaluated_results_in_dataset.append(final_entry)

                # Add to flat list only if evaluation resulted in a valid 'yes' or 'no' for metrics
                # And only if there wasn't a QA error or an Eval error
                if evaluation_result in ["yes", "no"] and not eval_error and dataset_name in self.target_dataset_names:
                    final_results_list_for_metrics.append(final_entry)

            dataset_eval_end_time = time.time()
            dataset_eval_duration = dataset_eval_end_time - dataset_eval_start_time
            intermediate_results_by_dataset[dataset_name]["results"] = (
                evaluated_results_in_dataset  # Update with eval results
            )
            intermediate_results_by_dataset[dataset_name]["duration_eval_seconds"] = (
                dataset_eval_duration
            )
            logging.info(
                f"    Finished evaluation for dataset {dataset_name} in {dataset_eval_duration:.2f} seconds."
            )

        overall_eval_end_time = time.time()
        overall_eval_duration = overall_eval_end_time - overall_eval_start_time
        logging.info(
            f"--- Finished Phase 2 (Evaluation) in {overall_eval_duration:.2f} seconds ---"
        )
        return (
            intermediate_results_by_dataset,
            overall_eval_duration,
            final_results_list_for_metrics,
        )

    def _calculate_and_print_metrics(
        self, final_results_list_for_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates and prints overall metrics based on a *pre-filtered* list
        containing only valid results from target datasets.
        """
        logging.info(f"\n--- Phase 3: Calculating Overall Metrics ---")
        if not final_results_list_for_metrics:
            logging.warning("  No valid results from target datasets available for metrics calculation.")
            # Return empty structure matching metrics output
            return {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1_score": 0.0,
                "true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0,
                "total_questions": 0, DATASET_METRIC_KEY: {}
            }

        # The input list is already filtered by _run_evaluation_phase
        logging.info(f"  Calculating metrics based on {len(final_results_list_for_metrics)} valid results from target datasets.")
        try:
            overall_metrics = calculate_metrics(final_results_list_for_metrics) # calculate_metrics uses 'dataset' key internally

            logging.info(f"\n--- Overall Evaluation Analysis (Based on Target Datasets) ---")
            for key, value in overall_metrics.items():
                if key == DATASET_METRIC_KEY: continue # Skip printing this sub-dict here
                metric_name = key.replace("_", " ").title()
                if isinstance(value, float):
                    logging.info(f"  {metric_name}: {value:.4f}")
                else:
                    logging.info(f"  {metric_name}: {value}")
            return overall_metrics
        except Exception as e:
            logging.error(f"  Error calculating overall metrics: {e}", exc_info=True)
            return {"metrics_calculation_error": str(e)} # Indicate error

    def _calculate_dataset_success_rates(
        self, evaluated_results_by_dataset: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculates the 'yes' success rate for each target dataset specified in the config.
        Only considers datasets present in `self.target_dataset_names`.
        """
        dataset_metrics = {}
        logging.debug("Calculating dataset self-evaluation success rates (for target datasets)...")

        # Iterate only through the datasets targeted by the current config
        for dataset_name in self.target_dataset_names:
            if dataset_name in evaluated_results_by_dataset:
                dataset_content = evaluated_results_by_dataset[dataset_name]
                results_list = dataset_content.get("results", [])

                if not isinstance(results_list, list):
                    logging.warning(
                        f"Dataset '{dataset_name}' has invalid 'results' format. Skipping rate calculation."
                    )
                    continue

                total_valid_for_rate = 0
                yes_count = 0
                for result_item in results_list:
                    # Consider item for rate if it has a judgment and no error
                    if isinstance(result_item, dict) and result_item.get("self_evaluation") in ["yes", "no"] and not result_item.get("eval_error"):
                        total_valid_for_rate += 1
                        if result_item.get("self_evaluation") == "yes":
                            yes_count += 1

                if total_valid_for_rate > 0:
                    rate = yes_count / total_valid_for_rate
                    dataset_metrics[dataset_name] = rate
                    logging.debug(f"  Dataset '{dataset_name}' success rate: {yes_count}/{total_valid_for_rate} = {rate:.4f}")
                else:
                    dataset_metrics[dataset_name] = 0.0 # Rate is 0 if no valid evaluated results
                    logging.debug(f"  Dataset '{dataset_name}': 0 valid results for rate calculation, rate = 0.0")
            else:
                 # Target dataset not found in results (e.g., QA failed entirely for it)
                 dataset_metrics[dataset_name] = 0.0
                 logging.debug(f"  Target Dataset '{dataset_name}' not found in evaluated results, rate = 0.0")

        return dataset_metrics

    def _process_single_combination(
        self,
        retrieval_algorithm: str,
        question_model_name: str,
        language: str,
        extension: str,
        file_identifier: str,
        base_collection_name: str,
        chunk_size: int,
        overlap_size: int,
        question_llm_connector: BaseLLMConnector,  # Accept initialized connector
        retriever: BaseRetriever,  # Accept initialized retriever (might need indexing)
    ):
        """
        Processes a single test combination, handling normal runs and reevaluation mode.
        """
        dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"

        logging.info(
            f"\n>>> Processing Combination: "
            f"File={file_identifier.upper()}, Model={question_model_name}, Algo={retrieval_algorithm}, "
            f"Chunk={chunk_size}, Overlap={overlap_size}, TopK={self.num_retrieved_docs} <<<"
        )

        # --- Reevaluation Handling ---
        pruned_previous_results = None
        questions_to_answer = None # List of dicts
        run_qa_phase = True # Assume we need to run QA unless told otherwise

        if self.enable_reevaluation:
            pruned_previous_results, questions_to_answer = self._handle_reevaluation(
                retrieval_algorithm, file_identifier, question_model_name,
                chunk_size, overlap_size, self.num_retrieved_docs
            )

            if pruned_previous_results is not None and questions_to_answer is not None:
                # Reevaluation is possible (file existed and was processed)
                if not questions_to_answer:
                    # No new questions to answer, just recalculate metrics and save
                    logging.info("Reevaluation: No new questions to answer. Recalculating metrics on pruned results and saving.")
                    run_qa_phase = False # Skip QA and Eval phases

                    # Recalculate metrics based *only* on target datasets within the pruned results
                    all_pruned_results_for_metrics = []
                    for ds_name, ds_content in pruned_previous_results.get("per_dataset_details", {}).items():
                        if ds_name in self.target_dataset_names: # Filter by target datasets
                            for res in ds_content.get("results", []):
                                if isinstance(res, dict) and res.get("self_evaluation") in ["yes", "no"] and not res.get("eval_error"):
                                    all_pruned_results_for_metrics.append(res)

                    overall_metrics = self._calculate_and_print_metrics(all_pruned_results_for_metrics)
                    if isinstance(overall_metrics, dict) and "metrics_calculation_error" not in overall_metrics:
                        # Calculate dataset rates based on all data, but only for target datasets
                        dataset_success_rates = self._calculate_dataset_success_rates(pruned_previous_results["per_dataset_details"])
                        overall_metrics[DATASET_METRIC_KEY] = dataset_success_rates
                        pruned_previous_results["overall_metrics"] = overall_metrics
                    else:
                        logging.error("Metrics calculation failed during reevaluation (no new questions). Saving without updated metrics.")
                        pruned_previous_results["overall_metrics"] = {"error": "Metrics recalculation failed", DATASET_METRIC_KEY: {}}

                    # Update parameters/timing notes?
                    if "test_run_parameters" not in pruned_previous_results: pruned_previous_results["test_run_parameters"] = {}
                    pruned_previous_results["test_run_parameters"]["language"] = language
                    pruned_previous_results["test_run_parameters"]["file_extension"] = extension
                    pruned_previous_results["test_run_parameters"]["file_tested"] = file_identifier
                    pruned_previous_results["test_run_parameters"]["reevaluation_run_type"] = "metrics_only"
                    pruned_previous_results["test_run_parameters"]["evaluator_model"] = self.config_loader.get_evaluator_model_name() # Update evaluator model potentially

                    self.result_manager.save_results(
                        results=pruned_previous_results, retrieval_algorithm=retrieval_algorithm, file_identifier=file_identifier,
                        question_model_name=question_model_name, chunk_size=chunk_size, overlap_size=overlap_size,
                        num_retrieved_docs=self.num_retrieved_docs
                    )
                    logging.info("Reevaluation complete (metrics only). Results saved.")
                    return # End processing for this combination

                # If we reach here, reevaluation is needed, questions_to_answer is not empty
                logging.info(f"Reevaluation: Proceeding to answer {len(questions_to_answer)} missing questions.")
                # run_qa_phase remains True

            # If pruned_previous_results is None or questions_to_answer is None,
            # it means reevaluation failed (e.g., file not found). Fall through to normal run.
            elif pruned_previous_results is None:
                 logging.info("Reevaluation enabled but no existing file found or load failed. Performing full run.")
                 questions_to_answer = None # Ensure override is not used in QA


        # --- Normal Run Check (only if not in successful reevaluation path) ---
        if run_qa_phase and not self.enable_reevaluation:
             if self.result_manager.load_previous_results(
                 retrieval_algorithm, file_identifier, question_model_name,
                 chunk_size, overlap_size, self.num_retrieved_docs
             ):
                 logging.info("Skipping combination: Results file already exists (Normal Run).")
                 return

        # --- Proceed with QA/Eval if needed ---
        final_results_to_save = {}
        if run_qa_phase:
            # --- Get Chroma Collection & Index Retriever (Common for both flows if proceeding) ---
            collection = self._get_chroma_collection(base_collection_name, chunk_size, overlap_size)
            if collection is None:
                logging.warning(f"Skipping combination: Failed to get required ChromaDB collection '{dynamic_collection_name}'.")
                return

            # --- Specific setup for Keyword or Hybrid Retriever Indexing ---
            if retrieval_algorithm == "keyword" or retrieval_algorithm == "hybrid":
                logging.info(f"{retrieval_algorithm.capitalize()} Algorithm: Indexing documents from '{collection.name}'...")
                try:
                    results = collection.get(include=["documents"])
                    if results and results.get("documents"):
                        document_chunks = results["documents"]
                        if document_chunks:
                            if isinstance(retriever, (KeywordRetriever, HybridRetriever)):
                                index_method = getattr(retriever, "build_keyword_index", getattr(retriever, "build_index", None))
                                if index_method:
                                     index_method(document_chunks)
                                     logging.info(f"Index built successfully for {len(document_chunks)} documents.")
                                else: raise AttributeError("No index building method found.")
                            else: raise TypeError(f"Invalid retriever type for indexing: {type(retriever).__name__}")
                        else:
                             logging.warning(f"Collection '{collection.name}' has no documents. Indexing empty.")
                             if isinstance(retriever, (KeywordRetriever, HybridRetriever)): # Build empty index
                                 index_method = getattr(retriever, "build_keyword_index", getattr(retriever, "build_index", None))
                                 if index_method: index_method([])
                    else:
                        logging.error(f"Failed to fetch documents from '{collection.name}' for indexing. Skipping.")
                        return
                except Exception as e:
                    logging.error(f"Error fetching/building index for {retrieval_algorithm}: {e}. Skipping.", exc_info=True)
                    return

            # --- Run QA Phase ---
            # Pass 'questions_to_answer' list if reevaluating, otherwise it's None (uses self.loaded_datasets)
            new_intermediate_results, qa_duration, qa_count = self._run_qa_phase(
                retriever=retriever, collection=collection, question_llm_connector=question_llm_connector,
                current_retrieval_algorithm=retrieval_algorithm, current_question_model_name=question_model_name,
                file_identifier=file_identifier, chunk_size=chunk_size, overlap_size=overlap_size,
                questions_to_process=questions_to_answer # Pass the list of missing questions
            )

            # --- Run Evaluation Phase ---
            # Evaluates only the results returned by the QA phase
            newly_evaluated_results, eval_duration, new_metric_results_list = self._run_evaluation_phase(
                intermediate_results_by_dataset=new_intermediate_results,
                total_questions_processed_in_qa=qa_count
            )

            # --- Combine Results & Calculate Metrics ---
            if self.enable_reevaluation and pruned_previous_results:
                # Reevaluation mode: Merge new results into pruned previous results
                logging.info("Reevaluation: Merging new results with pruned previous results using overwrite...")
                final_results_to_save = pruned_previous_results

                # Use dict for efficient overwrite: Convert existing results lists to dicts
                merged_details = {}
                for ds_name, ds_content in final_results_to_save.get("per_dataset_details", {}).items():
                    results_dict = { # Key: question string, Value: result dict
                        res["question"]: res
                        for res in ds_content.get("results", [])
                        if isinstance(res, dict) and "question" in res
                    }
                    merged_details[ds_name] = ds_content.copy() # Copy dataset-level keys (like duration if any)
                    merged_details[ds_name]["results"] = results_dict # Store as dict temporarily

                # Merge newly evaluated results into the dictionaries (overwrite/add)
                total_new_or_updated = 0
                for ds_name, new_content in newly_evaluated_results.items():
                    new_results_list = new_content.get("results", [])
                    if not new_results_list: continue # Skip if no new results for this dataset

                    if ds_name not in merged_details:
                        # Dataset is entirely new (wasn't in pruned_previous_results)
                        logging.info(f"  Adding new dataset '{ds_name}' from QA results.")
                        merged_details[ds_name] = new_content # Add the whole new structure
                        # Convert its results to dict for consistency IF needed later, but not strictly required here
                        # For simplicity, keep as list if just adding whole dataset
                        total_new_or_updated += len(new_results_list)
                    else:
                        # Dataset existed, merge/update individual questions into its dict
                        target_dict = merged_details[ds_name]["results"] # Get the results dict
                        updated_count = 0
                        added_count = 0
                        for new_res in new_results_list:
                             if isinstance(new_res, dict) and "question" in new_res:
                                 q_str = new_res["question"]
                                 if q_str in target_dict: updated_count += 1
                                 else: added_count += 1
                                 target_dict[q_str] = new_res # *** Overwrite or add ***
                                 total_new_or_updated += 1 # Count effective changes/additions
                        logging.info(f"  Merged results for dataset '{ds_name}': {added_count} added, {updated_count} updated.")

                # Convert results dictionaries back to lists before saving final structure
                for ds_name in merged_details:
                     if "results" in merged_details[ds_name] and isinstance(merged_details[ds_name]["results"], dict):
                           merged_details[ds_name]["results"] = list(merged_details[ds_name]["results"].values())

                # Update the final results structure
                final_results_to_save["per_dataset_details"] = merged_details
                logging.info(f"Total new/updated results merged: {total_new_or_updated}")
                # *** End Corrected Merge Logic ***


                # Recalculate metrics based on the *merged* results, filtered by target datasets
                logging.info("Reevaluation: Recalculating metrics on merged results...")
                all_merged_results_for_metrics = []

                for ds_name, ds_content in final_results_to_save.get("per_dataset_details", {}).items():
                    if ds_name in self.target_dataset_names:
                        for res in ds_content.get("results", []):
                            if isinstance(res, dict) and res.get("self_evaluation") in ["yes", "no"] and not res.get("eval_error"):
                                all_merged_results_for_metrics.append(res)

                overall_metrics = self._calculate_and_print_metrics(all_merged_results_for_metrics)
                if isinstance(overall_metrics, dict) and "metrics_calculation_error" not in overall_metrics:
                     # Calculate dataset rates based on combined data, only for target datasets
                     dataset_success_rates = self._calculate_dataset_success_rates(final_results_to_save["per_dataset_details"])
                     overall_metrics[DATASET_METRIC_KEY] = dataset_success_rates
                     final_results_to_save["overall_metrics"] = overall_metrics
                else:
                     logging.error("Metrics calculation failed during reevaluation merge. Saving without updated metrics.")
                     final_results_to_save["overall_metrics"] = {"error": "Combined metrics calculation failed", DATASET_METRIC_KEY: {}}

                # Update timing/parameters notes for reevaluation
                if "test_run_parameters" not in final_results_to_save: final_results_to_save["test_run_parameters"] = {}
                final_results_to_save["test_run_parameters"]["language"] = language
                final_results_to_save["test_run_parameters"]["file_extension"] = extension
                final_results_to_save["test_run_parameters"]["file_tested"] = file_identifier
                final_results_to_save["test_run_parameters"]["reevaluation_run_type"] = "qa_and_merge"
                final_results_to_save["test_run_parameters"]["reevaluation_qa_duration_seconds"] = qa_duration
                final_results_to_save["test_run_parameters"]["reevaluation_eval_duration_seconds"] = eval_duration
                final_results_to_save["test_run_parameters"]["evaluator_model"] = self.config_loader.get_evaluator_model_name() # Update evaluator potentially


            else:
                # Normal run: Calculate metrics on the newly evaluated results
                # new_metric_results_list already filtered for target datasets
                overall_metrics = self._calculate_and_print_metrics(new_metric_results_list)
                if isinstance(overall_metrics, dict) and "metrics_calculation_error" not in overall_metrics:
                     # Calculate dataset rates based on new data, only for target datasets
                     dataset_success_rates = self._calculate_dataset_success_rates(newly_evaluated_results)
                     overall_metrics[DATASET_METRIC_KEY] = dataset_success_rates
                else:
                     logging.error("Metrics calculation failed during normal run.")
                     if isinstance(overall_metrics, dict): # Keep error structure
                          overall_metrics[DATASET_METRIC_KEY] = {}
                     else:
                          overall_metrics = {"error": "Metrics calculation failed", DATASET_METRIC_KEY: {}}


                overall_duration = qa_duration + eval_duration
                # Prepare final structure for saving
                final_results_to_save = {
                    "test_run_parameters": {
                        "language": language,
                        "file_extension": extension,
                        "file_tested": file_identifier,
                        "question_model": question_model_name,
                        "evaluator_model": self.config_loader.get_evaluator_model_name(),
                        "retrieval_algorithm": retrieval_algorithm, "chunk_size": chunk_size,
                        "overlap_size": overlap_size, "num_retrieved_docs": self.num_retrieved_docs,
                        "chroma_collection_used": collection.name if collection else "N/A",
                    },
                    "overall_metrics": overall_metrics,
                    "timing": {
                        "overall_duration_seconds": overall_duration,
                        "duration_qa_phase_seconds": qa_duration,
                        "duration_eval_phase_seconds": eval_duration,
                    },
                    "per_dataset_details": newly_evaluated_results,
                }
        else:
            # This block is reached if run_qa_phase was False (reeval, no new Qs)
            # final_results_to_save should already be populated with pruned_previous_results
            # and updated metrics from the reevaluation block.
            pass

        # --- Save Results ---
        if final_results_to_save: # Ensure there's something to save
            logging.info(f"\n--- Saving Results ---")
            self.result_manager.save_results(
                results=final_results_to_save, retrieval_algorithm=retrieval_algorithm, file_identifier=file_identifier,
                question_model_name=question_model_name, chunk_size=chunk_size, overlap_size=overlap_size,
                num_retrieved_docs=self.num_retrieved_docs
            )
        else:
             logging.warning("No results generated or processed for saving in this combination.")


        logging.info(
            f"\n<<< Finished Combination: File={file_identifier.upper()}, Model={question_model_name}, Algo={retrieval_algorithm}, Chunk={chunk_size}, Overlap={overlap_size} <<<"
        )


    def run_tests(self):
        """
        Runs the full suite of tests based on the loaded configuration,
        iterating through all combinations.
        """
        logging.info("\n--- Starting Test Iterations ---")
        # Estimate total combinations for progress tracking
        # This is an estimation because not all file+extension combinations may exist
        estimated_combinations = (
            len(self.question_models_to_test)
            * len(self.chunk_sizes_to_test)
            * len(self.overlap_sizes_to_test)
            * len(self.retrieval_algorithms_to_test)
            * len(self.files_to_test)
            * len(self.file_extensions_to_test)
        )
        combination_count = 0

        # --- Start Iteration Loops ---
        for model_name in self.question_models_to_test:
            logging.info(
                f"\n{'=' * 20} Testing Question Model: {model_name} {'=' * 20}"
            )

            current_question_llm_connector: Optional[BaseLLMConnector] = None
            question_llm_type, _ = self.config_loader.get_llm_type_and_config(
                model_name
            )

            if not question_llm_type:
                logging.error(
                    f"Model '{model_name}' not found in any LLM configuration. Skipping this model."
                )
                continue

            try:
                current_question_llm_connector = (
                    self.llm_connector_manager.get_connector(
                        question_llm_type, model_name
                    )
                )
                logging.info(
                    f"Successfully initialized question connector for model: {model_name}"
                )
            except Exception as e:
                logging.error(
                    f"Error initializing question connector for model {model_name}: {e}. Skipping this model.",
                    exc_info=True,
                )
                continue

            for chunk_size in self.chunk_sizes_to_test:
                for overlap_size in self.overlap_sizes_to_test:
                    logging.info(
                        f"\n{'+' * 15} Testing Chunk/Overlap: CS={chunk_size}, OS={overlap_size} (Model: {model_name}) {'+' * 15}"
                    )

                    for algorithm in self.retrieval_algorithms_to_test:
                        logging.info(
                            f"\n{'-' * 10} Testing Retrieval Algorithm: {algorithm.upper()} (Model: {model_name}, CS={chunk_size}, OS={overlap_size}) {'-' * 10}"
                        )

                        for file_basename in self.files_to_test:
                            for extension in self.file_extensions_to_test:
                                manual_filepath = os.path.join(MANUALS_DIRECTORY, f"{file_basename}.{extension}")
                                if not os.path.isfile(manual_filepath):
                                    logging.debug(f"Skipping, file not found: {manual_filepath}")
                                    continue

                                combination_count += 1
                                logging.info(
                                    f"\n--- Running Combination {combination_count}/{estimated_combinations} (est.) ---"
                                )

                                # Define identifiers for this specific file
                                # Sanitize extension for use in names (e.g., 'xml.json' -> 'xml_json')
                                sanitized_ext = extension.replace('.', '_')
                                file_identifier = f"{file_basename}_{sanitized_ext}"
                                base_collection_name = file_identifier
                                # Calculate dynamic name here for retriever initialization
                                dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"

                                current_retriever: Optional[BaseRetriever] = None
                                try:
                                    # The 'initialize_retriever' function (to be refactored in the next step)
                                    # will now use the shared retriever for 'embedding' and 'hybrid' algorithms.
                                    current_retriever = initialize_retriever(
                                        retrieval_strategy_str=algorithm,
                                        embedding_model_config=self.embedding_model_config,
                                        chroma_client=self.chroma_client,
                                        collection_name=dynamic_collection_name,
                                        # Pass the shared instance to the initializer
                                        shared_embedding_retriever=self.shared_embedding_retriever
                                    )
                                    logging.info(f"Initialized retriever: {type(current_retriever).__name__} for algorithm '{algorithm}'")

                                except Exception as e:
                                    logging.error(
                                        f"Error initializing retriever for algorithm '{algorithm}': {e}. Skipping this combination.",
                                        exc_info=True,
                                    )
                                    continue

                                # Process this specific combination
                                self._process_single_combination(
                                    retrieval_algorithm=algorithm,
                                    question_model_name=model_name,
                                    language=file_basename,
                                    extension=extension,
                                    file_identifier=file_identifier,
                                    base_collection_name=base_collection_name,
                                    chunk_size=chunk_size,
                                    overlap_size=overlap_size,
                                    question_llm_connector=current_question_llm_connector,
                                    retriever=current_retriever,
                                )

        logging.info("\n--- All Test Combinations Completed ---")


def start_rag_tests(config_path: str = "config_fast.json", embedding_retriever: Optional[EmbeddingRetriever] = None):
    """
    Initializes and runs the RagTester.
    This function serves as the main entry point for external scripts like main.py.

    Args:
        config_path (str): The path to the configuration file.
        embedding_retriever (Optional[EmbeddingRetriever]): A pre-initialized and
            shared instance of the EmbeddingRetriever.
    """
    logging.info(
        f"--- Starting RAG tests via start_rag_tests (config: {config_path}) ---"
    )
    try:
        tester = RagTester(config_path=config_path, embedding_retriever=embedding_retriever)
        tester.run_tests()
        logging.info(f"--- RAG tests completed successfully via start_rag_tests ---")
        return True
    except Exception as e:
        logging.critical(
            f"--- RAG tests failed during execution initiated by start_rag_tests ---",
            exc_info=True,
        )
        raise e


# --- Main Execution ---
if __name__ == "__main__":
    config_to_test = "config_fast.json"

    logging.info(f"--- RAG Tester Script Start (Direct Execution) ---")
    logging.info(f"Using configuration file: {config_to_test}")

    # For standalone execution, we must create our own embedding model instance
    # This mirrors the behavior of main.py
    shared_embedding_retriever = None
    try:
        # --- Pre-computation Check & Standalone Model Initialization ---
        try:
            temp_config_loader = ConfigLoader(config_to_test)
            files_to_test_main = temp_config_loader.get_files_to_test()
            extensions_to_test_main = temp_config_loader.get_file_extensions_to_test()
            rag_params_main = temp_config_loader.get_rag_parameters()
            chunk_sizes_main = rag_params_main.get("chunk_sizes_to_test", "N/A")
            overlap_sizes_main = rag_params_main.get("overlap_sizes_to_test", "N/A")
            models_to_test_main = temp_config_loader.get_question_models_to_test()
            algos_to_test_main = temp_config_loader.get_retrieval_algorithms_to_test()

            logging.info(f"\nStarting RAG Tester for:")
            logging.info(f"  Files: {files_to_test_main}")
            logging.info(f"  Extensions: {extensions_to_test_main}")
            logging.info(f"  Question Models: {models_to_test_main}")
            logging.info(f"  Retrieval Algorithms: {algos_to_test_main}")
            logging.info(f"  Chunk Sizes: {chunk_sizes_main}")
            logging.info(f"  Overlap Sizes: {overlap_sizes_main}")
            
            logging.info("\n--- Standalone Mode: Initializing Shared Embedding Model ---")
            embedding_model_config = temp_config_loader.get_embedding_model_config()
            shared_embedding_retriever = EmbeddingRetriever(model_config=embedding_model_config)
            logging.info("--- Shared Embedding Model Initialized for Standalone Run ---")

            logging.info(
                f"\nIMPORTANT: This script will attempt to load ChromaDB collections specific to"
            )
            logging.info(
                f"           each configured file AND the chunk/overlap parameters being tested."
            )
            logging.info(
                f"           Collection name format: [file_basename]_[ext]_cs[chunk_size]_os[overlap_size]"
            )
            logging.info(
                f"           Ensure 'create_databases.py' has been run with"
            )
            logging.info(
                f"           combinations matching the 'chunk_sizes_to_test' and 'overlap_sizes_to_test'"
            )
            logging.info(
                f"           defined in '{config_to_test}' under 'rag_parameters'."
            )
        except Exception as config_ex:
            logging.error(
                f"Error loading configuration or initializing model for pre-check: {config_ex}", exc_info=True
            )
            raise

        # Pass the newly created model instance to the test runner
        start_rag_tests(
            config_path=config_to_test,
            embedding_retriever=shared_embedding_retriever
        )

    except FileNotFoundError as e:
        logging.critical(f"\nFATAL ERROR: Configuration file not found.")
        logging.critical(f"  Details: {e}")
    except ValueError as e:
        logging.critical(f"\nFATAL ERROR: Invalid or missing configuration.")
        logging.critical(f"  Details: {e}")
    except ImportError as e:
        if "rank_bm25" in str(e):
            logging.critical(f"\nFATAL ERROR: Missing dependency 'rank_bm25'.")
            logging.critical(f"  Please install it using: pip install rank-bm25")
        else:
            logging.critical(f"\nFATAL ERROR: Failed to import necessary modules.")
            logging.critical(f"  Details: {e}")
    except Exception as e:
        import traceback

        logging.critical(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.critical(f"FATAL ERROR: An unexpected error occurred during execution!")
        logging.critical(f"Error Type: {type(e).__name__}")
        logging.critical(f"Error Message: {e}")
        logging.critical("Traceback:")
        logging.critical(traceback.format_exc())
        logging.critical(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    finally:
        logging.info("--- RAG Tester Script End (Direct Execution) ---")
