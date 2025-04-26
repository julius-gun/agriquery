# --- Imports ---
import os
import sys
import json
import chromadb
import pathlib
import random
from typing import Optional, List, Dict, Any, Tuple

# --- Path Setup ---
# Adjust sys.path to find modules in the parent directory (RAG)
# This assumes the script is run from within the 'demonstration' directory
# or that the 'RAG' directory is added to PYTHONPATH externally.
current_script_path = pathlib.Path(__file__).resolve()
project_root = current_script_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# print(f"Adjusted sys.path. Project root assumed: {project_root}")
# print(f"Current sys.path: {sys.path}")


# --- Project Imports ---
# Now import modules relative to the project_root (RAG directory)
try:
    from utils.config_loader import ConfigLoader
    from llm_connectors.llm_connector_manager import LLMConnectorManager
    from llm_connectors.base_llm_connector import BaseLLMConnector
    from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
    from evaluation.evaluator import Evaluator
    print("Successfully imported project modules.")
except ImportError as e:
    print(f"ERROR: Failed to import necessary project modules: {e}")
    print(f"Project Root: {project_root}")
    print(f"Sys Path: {sys.path}")
    print("Ensure the script is run correctly relative to the 'RAG' directory or that paths are set up.")
    sys.exit(1) # Exit if imports fail

# --- Constants ---
DEFAULT_CONFIG_NAME = "config.json" # Relative to project_root
DEFAULT_DB_DIR_NAME = "chroma_db"   # Relative to project_root
DEFAULT_EVALUATOR_LLM_TYPE = "ollama" # Fallback

# --- Helper Functions ---

def find_llm_type(model_name: str, llm_configs: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Finds the type ('ollama', 'gemini', etc.) of a given model name from the config."""
    for type_key, models in llm_configs.items():
        if model_name in models:
            return type_key
    return None

def load_question_datasets(config_loader: ConfigLoader) -> Dict[str, List[Dict]]:
    """Loads question datasets specified in the config."""
    datasets = {}
    dataset_paths = config_loader.get_question_dataset_paths()
    if not dataset_paths:
        print("[WARN] No 'question_dataset_paths' found in configuration.")
        return datasets

    print("[INFO] Loading question datasets...")
    for name, rel_path in dataset_paths.items():
        abs_path = project_root / rel_path
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    datasets[name] = data
                    print(f"  Successfully loaded dataset '{name}' from {abs_path} ({len(data)} entries)")
                else:
                    print(f"  [WARN] Dataset '{name}' file ({abs_path}) does not contain a JSON list. Skipping.")
        except FileNotFoundError:
            print(f"  [ERROR] Dataset file not found for '{name}': {abs_path}. Skipping.")
        except json.JSONDecodeError:
            print(f"  [ERROR] Invalid JSON in dataset file for '{name}': {abs_path}. Skipping.")
        except Exception as e:
            print(f"  [ERROR] Failed to load dataset '{name}' from {abs_path}: {e}. Skipping.")
    return datasets

def select_question_from_dataset(datasets: Dict[str, List[Dict]]) -> Optional[Tuple[str, str, Optional[str]]]:
    """Allows user to select a dataset and returns a random question from it."""
    if not datasets:
        print("[WARN] No datasets available to select from.")
        return None

    print("\nAvailable datasets:")
    dataset_names = list(datasets.keys())
    for i, name in enumerate(dataset_names):
        print(f"  {i + 1}: {name} ({len(datasets[name])} questions)")

    while True:
        try:
            choice = input(f"Select dataset number (1-{len(dataset_names)}) or [C]ancel: ").upper()
            if choice == 'C':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(dataset_names):
                selected_name = dataset_names[choice_idx]
                selected_dataset = datasets[selected_name]
                if not selected_dataset:
                    print(f"[WARN] Dataset '{selected_name}' is empty. Please choose another.")
                    continue # Re-prompt dataset choice

                # Select a random question entry
                random_entry = random.choice(selected_dataset)

                # Extract question and expected answer (handle missing keys)
                question = random_entry.get("question")
                expected_answer = random_entry.get("answer") # Use .get() for safety

                if not question:
                     print(f"[WARN] Selected entry in '{selected_name}' is missing the 'question' key. Trying again...")
                     # Potentially implement logic to retry random selection a few times
                     # For simplicity now, we just inform and let the user try again if needed
                     return None # Or continue the outer loop

                # Return question, expected answer (can be None), and dataset name
                return question, expected_answer, selected_name
            else:
                print(f"[WARN] Invalid choice. Please enter a number between 1 and {len(dataset_names)} or C.")
        except ValueError:
            print("[WARN] Invalid input. Please enter a number or C.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during dataset selection: {e}")
            return None # Exit selection on unexpected error

def get_user_question() -> str:
    """Prompts the user to enter a question."""
    while True:
        question = input("Enter your question: ").strip()
        if question:
            return question
        else:
            print("[WARN] Question cannot be empty.")

def retrieve_context(
    question: str,
    collection: chromadb.Collection,
    retriever: EmbeddingRetriever,
    top_k: int
) -> Tuple[List[str], List[float], str]:
    """Retrieves context chunks from ChromaDB for a given question."""
    print("[INFO] Vectorizing question...")
    # EmbeddingRetriever.vectorize_text returns List[List[float]]
    question_embedding_list = retriever.vectorize_text(question)
    if not question_embedding_list or not isinstance(question_embedding_list[0], list):
         raise ValueError("Failed to vectorize question or unexpected format received.")
    query_vector = question_embedding_list[0] # Extract the single embedding vector
    print("[INFO] Question vectorized successfully.")

    print(f"[INFO] Retrieving top {top_k} relevant documents from collection '{collection.name}'...")
    results = collection.query(
        query_embeddings=[query_vector], # Query expects List[List[float]]
        n_results=top_k,
        include=["documents", "distances"]
    )
    print("[INFO] Retrieval complete.")

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_distances = results.get("distances", [[]])[0]
    context_string = "\n\n".join(retrieved_docs)

    return retrieved_docs, retrieved_distances, context_string

# --- Placeholder functions for LLM Query and Evaluation ---
# These will be implemented in the next steps

def query_llm(
    question: str,
    context: str,
    llm_connector: BaseLLMConnector,
    question_prompt_template: str
) -> Optional[str]:
    """Queries the LLM with the question and context, showing the prompt."""
    print("\n[INFO] Formatting question prompt...")
    try:
        formatted_prompt = question_prompt_template.format(
            context=context, question=question
        )

        print("\n" + "=" * 20 + " LLM Query Prompt " + "=" * 20)
        print(formatted_prompt)
        print("=" * (len("=" * 20 + " LLM Query Prompt " + "=" * 20))) # Match header

        print(f"\n[INFO] Sending request to LLM: {llm_connector.model_name}...")
        llm_answer = llm_connector.invoke(formatted_prompt)
        print("[INFO] Received response from LLM.")

        # Display Answer (moved here from main loop)
        print("\n--- LLM Answer ---")
        print(llm_answer)
        print("------------------")
        return llm_answer.strip() # Return stripped answer

    except KeyError as e:
        print(f"[ERROR] Missing placeholder in question prompt template: {e}")
        print("  Template:", question_prompt_template)
        return None
    except Exception as e:
        print(f"[ERROR] Error during LLM invocation: {e}")
        return None

def evaluate_llm_answer(
    question: str,
    llm_answer: str,
    expected_answer: str,
    evaluator: Evaluator,
    evaluation_prompt_template: str # Pass the template content here
) -> Optional[str]:
    """Evaluates the LLM's answer against the expected answer, showing the prompt."""
    print("\n[INFO] Formatting evaluation prompt...")
    try:
        # Format the prompt *before* calling the evaluator for display purposes
        # The Evaluator class itself takes the template at init, but we need to show the formatted version
        formatted_eval_prompt = evaluation_prompt_template.format(
            question=question,
            model_answer=llm_answer,
            expected_answer=expected_answer
        )

        print("\n" + "=" * 20 + " Evaluation Prompt " + "=" * 20)
        print(formatted_eval_prompt)
        print("=" * (len("=" * 20 + " Evaluation Prompt " + "=" * 20))) # Match header

        print(f"\n[INFO] Sending request to Evaluator LLM: {evaluator.evaluator_model_connector.model_name}...")
        # The evaluator object uses its *own* template internally when evaluate_answer is called
        evaluation_result = evaluator.evaluate_answer(
            question=question,
            model_answer=llm_answer,
            expected_answer=expected_answer,
        )
        print("[INFO] Received response from Evaluator LLM.")

        # Display Evaluation Result (moved here from main loop)
        print(f"\n--- Evaluation Result (Expected: '{expected_answer}') ---")
        print(f"Judgment: {evaluation_result}")
        print("----------------------------------------------------")
        return evaluation_result.strip().lower() # Return normalized result

    except KeyError as e:
        print(f"[ERROR] Missing placeholder in evaluation prompt template: {e}")
        print("  Template:", evaluation_prompt_template)
        return None
    except Exception as e:
        print(f"[ERROR] Error during evaluation invocation: {e}")
        return None

# --- Main Application Logic ---

def main():
    print("--- Interactive RAG Demo ---")

    # 1. Load Configuration
    config_path = project_root / DEFAULT_CONFIG_NAME
    db_path = project_root / DEFAULT_DB_DIR_NAME
    try:
        config_loader = ConfigLoader(str(config_path))
        config = config_loader.config
        rag_params = config_loader.get_rag_parameters()
        language_configs = config.get("language_configs", [])
        llm_models_config = config.get("llm_models", {})

        # Determine Language, Chunk Size, Overlap, Top K (using first from config as default)
        if not language_configs: raise ValueError("No 'language_configs' found in configuration.")
        lang_config = language_configs[0] # Use the first language defined
        target_language = lang_config.get("language")
        collection_base_name = lang_config.get("collection_base_name")
        if not target_language or not collection_base_name: raise ValueError("First language_config missing 'language' or 'collection_base_name'.")

        chunk_sizes = rag_params.get("chunk_sizes_to_test", [])
        if not chunk_sizes: raise ValueError("'chunk_sizes_to_test' not found or empty in rag_parameters.")
        chunk_size = chunk_sizes[0]

        overlap_sizes = rag_params.get("overlap_sizes_to_test", [])
        if not overlap_sizes: raise ValueError("'overlap_sizes_to_test' not found or empty in rag_parameters.")
        overlap_size = overlap_sizes[0]

        top_k = rag_params.get("num_retrieved_docs", 3) # Default to 3 if not specified

        dynamic_collection_name = f"{collection_base_name}_cs{chunk_size}_os{overlap_size}"
        print(f"[INFO] Using Collection: '{dynamic_collection_name}' (Lang: {target_language}, CS: {chunk_size}, OS: {overlap_size}, TopK: {top_k})")

    except Exception as e:
        print(f"[ERROR] Failed to load configuration or determine parameters: {e}")
        sys.exit(1)

    # 2. Load Question Datasets
    datasets = load_question_datasets(config_loader)
    if not datasets:
        print("[WARN] No question datasets loaded. Only manual questions possible.")
    else:
        print(f"[INFO] Loaded question datasets: {list(datasets.keys())}")

    # 3. Initialize ChromaDB and Retriever
    collection = None
    retriever = None
    try:
        if not db_path.exists() or not db_path.is_dir():
            raise FileNotFoundError(f"ChromaDB directory not found: {db_path}")

        chroma_client = chromadb.PersistentClient(path=str(db_path))
        print("[INFO] ChromaDB client initialized.")

        print(f"[INFO] Attempting to get collection: '{dynamic_collection_name}'")
        # Note: get_collection doesn't need embedding_function if it already exists with one.
        # If creating, it would be needed. For demo, assume it exists.
        collection = chroma_client.get_collection(name=dynamic_collection_name)
        print(f"[INFO] Successfully connected to collection '{dynamic_collection_name}'. It contains {collection.count()} items.")

        print("[INFO] Initializing embedding retriever...")
        # TODO: Make embedding model configurable if needed
        retriever = EmbeddingRetriever() # Uses default model
        print("[INFO] Embedding retriever initialized.")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        # Catch potential ChromaDB errors (e.g., collection not found)
        print(f"[ERROR] Failed to initialize ChromaDB/Retriever: {e}")
        # Attempt to list available collections for debugging help
        try:
            available_collections = [col.name for col in chroma_client.list_collections()]
            print(f"Available collections: {available_collections}")
        except Exception as list_e:
            print(f"[WARN] Could not list collections after error: {list_e}")
        sys.exit(1)


    # 4. Initialize LLM Manager
    llm_connector_manager = LLMConnectorManager(llm_models_config)
    print("[INFO] LLM Connector Manager initialized.")

    # 5. Initialize Evaluator (Lazy initialization inside the loop)
    evaluator: Optional[Evaluator] = None # Type hint for clarity
    evaluator_initialized = False
    evaluation_prompt_template_content = "" # Store loaded template content
    print("[INFO] Evaluator will be initialized if needed.")


    # --- Interactive Loop ---
    while True:
        question = None
        expected_answer = None
        dataset_name = None
        llm_answer = None
        context_string = "" # Reset context for each loop

        # --- Get Question ---
        action = input("\nChoose action: [D]ataset question, [N]ew question, [Q]uit: ").upper()

        if action == 'D':
            if not datasets:
                print("[WARN] No datasets loaded. Please enter a question manually.")
                continue
            selection = select_question_from_dataset(datasets)
            if selection:
                question, expected_answer, dataset_name = selection
                print(f"\nSelected from '{dataset_name}':")
                print(f"  Question: {question}")
                # Don't print expected answer here, only if evaluating later
            else:
                # User cancelled selection or error occurred in selection
                continue
        elif action == 'N':
            question = get_user_question()
            expected_answer = None # No automatic expected answer
            dataset_name = "user_input"
            print(f"\nUsing user question: {question}")
        elif action == 'Q':
            break # Exit the loop
        else:
            print("[WARN] Invalid action. Please choose D, N, or Q.")
            continue

        # --- Retrieve Context ---
        if question and collection and retriever:
            try:
                retrieved_docs, retrieved_distances, context_string = retrieve_context(
                    question, collection, retriever, top_k
                )
                print("\n" + "=" * 20 + " Retrieved Context " + "=" * 20)
                if not retrieved_docs:
                    print("No relevant documents found.")
                else:
                    for i, (doc, dist) in enumerate(zip(retrieved_docs, retrieved_distances)):
                        print(f"\n--- Document {i + 1} (Distance: {dist:.4f}) ---")
                        print(doc)
                        print("-" * (len(f"--- Document {i + 1} (Distance: {dist:.4f}) ---")))
                print("=" * (len("=" * 20 + " Retrieved Context " + "=" * 20))) # Match header length
            except Exception as e:
                print(f"[ERROR] Failed to retrieve context: {e}")
                context_string = "" # Ensure context is empty on error
        else:
            print("[WARN] Skipping context retrieval (missing question, collection, or retriever).")
            context_string = ""


        # --- Optional: Query LLM ---
        if context_string: # Only proceed if context was retrieved
            query_choice = input("\nQuery an LLM with this context? [Y/N]: ").upper()
            if query_choice == 'Y':
                # Show available models for user convenience
                print("Available LLM models (from config):")
                available_models = []
                for type_key, models in llm_models_config.items():
                     for model_key in models.keys():
                         print(f"  - {model_key} (type: {type_key})")
                         available_models.append(model_key)

                llm_model_to_use = input("Enter LLM model name to use: ").strip()

                if llm_model_to_use not in available_models:
                     print(f"[WARN] Model '{llm_model_to_use}' not found in config's known models. Attempting anyway...")
                     # We might still try if the user knows a model exists (e.g., Ollama served model not in config)
                     # However, finding its type will be harder. Let's rely on find_llm_type for now.

                try:
                    llm_type = find_llm_type(llm_model_to_use, llm_models_config)
                    if not llm_type:
                        # If not found in config, maybe make a guess or ask user? For now, error out.
                        raise ValueError(f"Could not determine LLM type for '{llm_model_to_use}' from config.")

                    print(f"[INFO] Determined LLM type: {llm_type}")
                    llm_connector = llm_connector_manager.get_connector(llm_type, llm_model_to_use)
                    print(f"[INFO] LLM connector obtained for {llm_model_to_use}.")

                    question_prompt_template_content = config_loader.load_prompt_template("question_prompt")

                    # Call the implemented query function
                    llm_answer = query_llm(question, context_string, llm_connector, question_prompt_template_content)
                    # Display is handled within query_llm

                    if not llm_answer:
                        print("[WARN] Failed to get answer from LLM.")

                except FileNotFoundError as e:
                    print(f"[ERROR] Prompt file error: {e}")
                except ValueError as e:
                    print(f"[ERROR] Configuration or LLM setup error: {e}")
                except Exception as e:
                    print(f"[ERROR] Error during LLM interaction: {e}")
                    llm_answer = None # Ensure answer is None if error occurred

        # --- Optional: Evaluate Answer ---
        if llm_answer: # Only if we got an LLM answer
            eval_choice = input("\nEvaluate the LLM answer? [Y/N]: ").upper()
            if eval_choice == 'Y':
                # Determine expected answer
                current_expected_answer = expected_answer # Use dataset answer if available
                if current_expected_answer is None: # If it wasn't from a dataset or dataset lacked 'answer'
                    current_expected_answer = input("Please provide the expected answer for evaluation: ").strip()

                if current_expected_answer: # Ensure we have an expected answer
                    try:
                        # Initialize Evaluator on first use
                        if not evaluator_initialized:
                            print("[INFO] Initializing Evaluator...")
                            evaluator_model_name = config_loader.get_evaluator_model_name()
                            if not evaluator_model_name: raise ValueError("Evaluator model name not found in config.")

                            evaluator_llm_type = find_llm_type(evaluator_model_name, llm_models_config)
                            if not evaluator_llm_type:
                                evaluator_llm_type = DEFAULT_EVALUATOR_LLM_TYPE
                                print(f"[WARN] Could not determine type for evaluator '{evaluator_model_name}'. Assuming '{evaluator_llm_type}'.")

                            evaluator_llm_connector = llm_connector_manager.get_connector(evaluator_llm_type, evaluator_model_name)
                            # Load the template content ONCE during initialization
                            evaluation_prompt_template_content = config_loader.load_prompt_template("evaluation_prompt")

                            evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template_content)
                            evaluator_initialized = True
                            print(f"[INFO] Evaluator initialized with model: {evaluator_model_name}")
                        elif not evaluator: # Should not happen if initialized is True, but safety check
                             raise RuntimeError("Evaluator should have been initialized but is not.")
                        elif not evaluation_prompt_template_content: # Safety check if template wasn't loaded
                             raise RuntimeError("Evaluation prompt template content not loaded.")


                        # Call the actual evaluation function (which needs implementation)
                        eval_result = evaluate_llm_answer(
                            question, llm_answer, current_expected_answer,
                            evaluator, evaluation_prompt_template_content # Pass the loaded template content
                        )
                        # Display is handled within evaluate_llm_answer

                        if not eval_result:
                            print("[WARN] Failed to get evaluation result.")

                    except FileNotFoundError as e:
                        print(f"[ERROR] Evaluation prompt file error: {e}")
                    except ValueError as e:
                        print(f"[ERROR] Configuration or Evaluator setup error: {e}")
                    except Exception as e:
                        print(f"[ERROR] Error during evaluation: {e}")
                else:
                    print("[INFO] Skipping evaluation as no expected answer was provided.")

        # --- Loop Continuation ---
        # The loop continues automatically unless 'Q' was chosen at the start

    print("\n--- Demo Finished ---") # This line is now reachable after the loop breaks

if __name__ == "__main__":
    main()