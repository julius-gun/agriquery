import argparse
import os
import sys
import json
import chromadb
import pathlib
import logging
from typing import Optional, List, Dict, Any

# --- Path Setup ---
# Add the project root directory (p_llm_manual/RAG) to the Python path
# This allows finding modules like utils, llm_connectors, etc.
current_script_path = pathlib.Path(__file__).resolve()
project_root = current_script_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Adjustment ---

# --- Project Imports ---
try:
    from utils.config_loader import ConfigLoader
    from llm_connectors.llm_connector_manager import LLMConnectorManager
    from llm_connectors.base_llm_connector import BaseLLMConnector
    from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
    from evaluation.evaluator import Evaluator
except ImportError as e:
    print(f"Error: Failed to import necessary project modules: {e}")
    print(f"Project Root: {project_root}")
    print(f"Sys Path: {sys.path}")
    print(
        "Ensure the script is run from the 'p_llm_manual/RAG' directory or that the path setup is correct."
    )
    sys.exit(1)

# --- Configure Logging ---
# Basic logging for info and errors
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Suppress excessive logging from underlying libraries if needed
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# --- Constants ---
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_DB_DIR_NAME = "chroma_db"
DEFAULT_EVALUATOR_LLM_TYPE = "ollama"  # Fallback if type cannot be determined


def find_llm_type(
    model_name: str, llm_configs: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """Finds the type ('ollama', 'gemini', etc.) of a given model name from the config."""
    for type_key, models in llm_configs.items():
        if model_name in models:
            return type_key
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question, retrieve context, optionally query an LLM and evaluate."
    )
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help=f"Path to the configuration file (relative to project root). Default: {DEFAULT_CONFIG_NAME}",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Target language (e.g., 'english', 'german'). If not provided, uses the first language in config.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size used for the database collection. If not provided, uses the first from config's rag_parameters.",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        help="Overlap size used for the database collection. If not provided, uses the first from config's rag_parameters.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of documents to retrieve. If not provided, uses value from config's rag_parameters.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="Name of the LLM model (key from config) to ask the question with context.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If set, evaluate the LLM's answer (requires --llm and prompts for expected answer).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_DIR_NAME,
        help=f"Path to the ChromaDB persistence directory (relative to project root). Default: {DEFAULT_DB_DIR_NAME}",
    )

    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config_path = project_root / args.config
    db_path = project_root / args.db_path
    logging.info(f"Using configuration file: {config_path}")
    logging.info(f"Using ChromaDB path: {db_path}")

    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        config_loader = ConfigLoader(str(config_path))
        config = config_loader.config
        rag_params = config_loader.get_rag_parameters()
        language_configs = config.get("language_configs", [])
        llm_models_config = config.get("llm_models", {})
    except Exception as e:
        logging.error(f"Error loading configuration: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Determine Parameters ---
    # Language
    target_language = args.language
    lang_config = None
    if target_language:
        lang_config = next(
            (lc for lc in language_configs if lc.get("language") == target_language),
            None,
        )
        if not lang_config:
            logging.error(
                f"Language '{target_language}' not found in configuration's language_configs."
            )
            sys.exit(1)
    elif language_configs:
        lang_config = language_configs[0]
        target_language = lang_config.get("language")
        logging.info(
            f"No language specified, using the first from config: '{target_language}'"
        )
    else:
        logging.error(
            "No language specified and no language_configs found in configuration."
        )
        sys.exit(1)

    collection_base_name = lang_config.get("collection_base_name")
    if not collection_base_name:
        logging.error(
            f"Missing 'collection_base_name' for language '{target_language}' in config."
        )
        sys.exit(1)

    # Chunk Size
    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_sizes_in_config = rag_params.get("chunk_sizes_to_test", [])
        if chunk_sizes_in_config:
            chunk_size = chunk_sizes_in_config[0]
            logging.info(
                f"No chunk size specified, using first from config: {chunk_size}"
            )
        else:
            logging.error(
                "No chunk size specified and 'chunk_sizes_to_test' not found or empty in config's rag_parameters."
            )
            sys.exit(1)

    # Overlap Size
    overlap_size = args.overlap_size
    if overlap_size is None:
        overlap_sizes_in_config = rag_params.get("overlap_sizes_to_test", [])
        if overlap_sizes_in_config:
            overlap_size = overlap_sizes_in_config[0]
            logging.info(
                f"No overlap size specified, using first from config: {overlap_size}"
            )
        else:
            logging.error(
                "No overlap size specified and 'overlap_sizes_to_test' not found or empty in config's rag_parameters."
            )
            sys.exit(1)

    # Top K
    top_k = args.top_k
    if top_k is None:
        top_k = rag_params.get("num_retrieved_docs")
        if top_k is None:
            logging.warning(
                "No top-k specified and 'num_retrieved_docs' not found in config. Defaulting to 3."
            )
            top_k = 3
        else:
            logging.info(f"Using top-k from config: {top_k}")

    # --- 3. Construct Collection Name ---
    dynamic_collection_name = f"{collection_base_name}_cs{chunk_size}_os{overlap_size}"
    logging.info(f"Attempting to use ChromaDB collection: '{dynamic_collection_name}'")

    # --- 4. Initialize ChromaDB and Get Collection ---
    if not db_path.exists() or not db_path.is_dir():
        logging.error(f"ChromaDB directory not found: {db_path}")
        logging.error(
            "Please ensure the database has been created using 'create_databases.py'."
        )
        sys.exit(1)

    try:
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        # Check if collection exists before getting it to provide a clearer error
        existing_collections = chroma_client.list_collections()
        if dynamic_collection_name not in existing_collections:
             logging.error(f"Collection '{dynamic_collection_name}' not found in the database at {db_path}.")
             # Convert list to string for cleaner logging, especially if long
             collections_str = ", ".join(existing_collections) if existing_collections else "None"
             logging.error(f"Available collections: [{collections_str}]")
             logging.error("Ensure the collection was created with the correct language, chunk size, and overlap.")
             sys.exit(1)

        # Now get the collection (should succeed)
        # Note: Embedding function isn't strictly needed here if we only query by vector
        collection = chroma_client.get_collection(name=dynamic_collection_name)
        logging.info(
            f"Successfully connected to collection '{dynamic_collection_name}'. It contains {collection.count()} items."
        )

    except Exception as e:
        logging.error(
            f"Error connecting to ChromaDB or getting collection '{dynamic_collection_name}': {e}",
            exc_info=True,
        )
        sys.exit(1)

    # --- 5. Initialize Retriever and Vectorize Question ---
    try:
        # Using default embedding model assumed by create_databases.py
        # TODO: Make embedding model configurable if needed
        retriever = EmbeddingRetriever()
        logging.info("Vectorizing question...")
        question_embedding = retriever.vectorize_text(args.question)
        # Ensure it's a flat list for ChromaDB query
        if (
            isinstance(question_embedding, list)
            and len(question_embedding) == 1
            and isinstance(question_embedding[0], list)
        ):
            query_vector = question_embedding[0]
        else:
            # Should not happen with current EmbeddingRetriever, but good practice
            logging.error("Unexpected embedding format received from retriever.")
            sys.exit(1)

    except Exception as e:
        logging.error(
            f"Error initializing retriever or vectorizing question: {e}", exc_info=True
        )
        sys.exit(1)

    # --- 6. Retrieve Context ---
    try:
        logging.info(f"Retrieving top {top_k} relevant documents...")
        results = collection.query(
            query_embeddings=[query_vector],  # Query expects List[List[float]]
            n_results=top_k,
            include=["documents", "distances"],  # Include distances for info
        )
    except Exception as e:
        logging.error(
            f"Error querying collection '{dynamic_collection_name}': {e}", exc_info=True
        )
        sys.exit(1)

    # --- 7. Display Context ---
    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_distances = results.get("distances", [[]])[0]

    print("\n" + "=" * 20 + " Retrieved Context " + "=" * 20)
    if not retrieved_docs:
        print("No relevant documents found.")
    else:
        for i, (doc, dist) in enumerate(zip(retrieved_docs, retrieved_distances)):
            print(f"\n--- Document {i + 1} (Distance: {dist:.4f}) ---")
            print(doc)
            print(
                "-" * (len(f"--- Document {i + 1} (Distance: {dist:.4f}) ---"))
            )  # Match length
    print("=" * 59)  # Match length of header

    context_string = "\n\n".join(retrieved_docs)

    # --- 8. LLM Interaction (Optional) ---
    llm_answer = None
    if args.llm:
        print(f"\n--- Querying LLM: {args.llm} ---")
        try:
            # Initialize LLM Manager
            llm_connector_manager = LLMConnectorManager(llm_models_config)

            # Determine LLM type
            llm_type = find_llm_type(args.llm, llm_models_config)
            if not llm_type:
                # Try fallback if not found directly (e.g., user provided alias not matching key exactly)
                # This part might need refinement depending on how robust alias handling should be
                logging.warning(
                    f"Could not directly determine type for LLM '{args.llm}'. Checking configurations..."
                )
                # A simple check: does any config *value* contain the name? Less reliable.
                found_config = None
                for type_key, models in llm_models_config.items():
                    for model_key, config_details in models.items():
                        if config_details.get("name") == args.llm:
                            llm_type = type_key
                            args.llm = model_key  # Use the config key now
                            found_config = config_details
                            logging.info(
                                f"Found matching config under type '{llm_type}' with key '{model_key}'."
                            )
                            break
                    if llm_type:
                        break

                if not llm_type:
                    logging.error(
                        f"LLM '{args.llm}' not found in any configuration under 'llm_models'. Cannot proceed."
                    )
                    sys.exit(1)

            # Get LLM Connector
            llm_connector = llm_connector_manager.get_connector(llm_type, args.llm)

            # Load Question Prompt
            question_prompt_template = config_loader.load_prompt_template(
                "question_prompt"
            )

            # Format Prompt
            formatted_prompt = question_prompt_template.format(
                context=context_string, question=args.question
            )

            # Invoke LLM
            logging.info("Sending request to LLM...")
            llm_answer = llm_connector.invoke(formatted_prompt)
            logging.info("Received response from LLM.")

            # Display Answer
            print("\n--- LLM Answer ---")
            print(llm_answer)
            print("------------------")

        except FileNotFoundError as e:
            logging.error(f"Prompt file error: {e}")
        except ValueError as e:  # Catch errors from get_connector or config issues
            logging.error(f"Configuration or LLM setup error: {e}")
        except Exception as e:
            logging.error(f"Error during LLM interaction: {e}", exc_info=True)
            # Continue to evaluation if requested, but answer will be None

    # --- 9. Evaluation (Optional) ---
    if args.evaluate:
        if not args.llm:
            logging.warning(
                "Evaluation requested (--evaluate) but no LLM was specified (--llm). Skipping evaluation."
            )
        elif llm_answer is None:
            logging.warning(
                "Evaluation requested, but failed to get an answer from the LLM. Skipping evaluation."
            )
        else:
            print("\n--- Evaluating LLM Answer ---")
            try:
                expected_answer = input(
                    "Please provide the expected answer for evaluation: "
                )

                # Initialize Evaluator
                evaluator_model_name = config_loader.get_evaluator_model_name()
                if not evaluator_model_name:
                    logging.error(
                        "Evaluator model name not found in config. Cannot evaluate."
                    )
                else:
                    evaluator_llm_type = find_llm_type(
                        evaluator_model_name, llm_models_config
                    )
                    if not evaluator_llm_type:
                        evaluator_llm_type = DEFAULT_EVALUATOR_LLM_TYPE
                        logging.warning(
                            f"Could not determine type for evaluator '{evaluator_model_name}'. Assuming '{evaluator_llm_type}'."
                        )

                    # Need LLM manager again if not created before
                    if "llm_connector_manager" not in locals():
                        llm_connector_manager = LLMConnectorManager(llm_models_config)

                    evaluator_llm_connector = llm_connector_manager.get_connector(
                        evaluator_llm_type, evaluator_model_name
                    )
                    evaluation_prompt_template = config_loader.load_prompt_template(
                        "evaluation_prompt"
                    )

                    evaluator = Evaluator(
                        evaluator_llm_connector, evaluation_prompt_template
                    )

                    # Perform Evaluation
                    logging.info("Sending request to Evaluator LLM...")
                    evaluation_result = evaluator.evaluate_answer(
                        question=args.question,
                        model_answer=llm_answer,
                        expected_answer=expected_answer,
                    )
                    logging.info("Received response from Evaluator LLM.")

                    # Display Evaluation Result
                    print(f"\n--- Evaluation Result (Expected: {expected_answer}) ---")
                    print(f"Judgment: {evaluation_result}")
                    print("----------------------------------------------------")

            except FileNotFoundError as e:
                logging.error(f"Evaluation prompt file error: {e}")
            except ValueError as e:  # Catch errors from get_connector or config issues
                logging.error(f"Configuration or Evaluator setup error: {e}")
            except Exception as e:
                logging.error(f"Error during evaluation: {e}", exc_info=True)

    print("\nScript finished.")


if __name__ == "__main__":
    main()
