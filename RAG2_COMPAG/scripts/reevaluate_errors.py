import os
import json
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple

# Disable Hugging Face telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# --- Path Setup ---
# This script is inside the 'utils' directory.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Project root is the parent directory of 'utils'
project_root_dir = os.path.dirname(current_script_dir)
# Add project root to sys.path to allow imports from other packages like llm_connectors, evaluation
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# --- Project Imports ---
try:
    # Can still use absolute-like imports from project root perspective
    from utils.config_loader import ConfigLoader
    from llm_connectors.llm_connector_manager import LLMConnectorManager
    from llm_connectors.base_llm_connector import BaseLLMConnector
    from evaluation.evaluator import Evaluator
    from evaluation.metrics import calculate_metrics
except ImportError as e:
    print(f"Error: Failed to import necessary project modules: {e}")
    print("Please ensure the script is run from a location where it can find 'utils', 'llm_connectors', 'evaluation' relative to the project root.")
    print(f"Calculated project root: {project_root_dir}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define paths relative to the project root
DEFAULT_RESULTS_DIR = os.path.join(project_root_dir, "results")
DEFAULT_CONFIG_PATH = os.path.join(project_root_dir, "config.json")
# Define default LLM type if not specified in config for evaluator (though config should have it)
DEFAULT_EVALUATOR_LLM_TYPE = "ollama"

def initialize_components(config_path: str) -> Tuple[Optional[Evaluator], Optional[ConfigLoader]]:
    """Loads config and initializes the Evaluator component."""
    logging.info(f"--- Initializing components using config: {config_path} ---")
    try:
        # ConfigLoader path is now absolute or relative to project root
        config_loader = ConfigLoader(config_path)
        config = config_loader.config

        # Initialize LLM Manager
        llm_connector_manager = LLMConnectorManager(config.get("llm_models", {}))

        # Get Evaluator Config
        evaluator_model_name = config_loader.get_evaluator_model_name()
        if not evaluator_model_name:
            logging.error("Evaluator model name not found in config. Cannot proceed.")
            return None, None

        # Determine evaluator LLM type (look for it in llm_models first, then default)
        evaluator_llm_type = None
        for type_key, models in config.get("llm_models", {}).items():
            if evaluator_model_name in models:
                evaluator_llm_type = type_key
                break
        if not evaluator_llm_type:
            evaluator_llm_type = DEFAULT_EVALUATOR_LLM_TYPE # Fallback
            logging.warning(f"Could not determine LLM type for evaluator '{evaluator_model_name}' from config. Assuming '{evaluator_llm_type}'.")


        # Get Evaluator LLM Connector
        evaluator_llm_connector = llm_connector_manager.get_connector(evaluator_llm_type, evaluator_model_name)

        # Load Evaluation Prompt Template
        evaluation_prompt_template = config_loader.load_prompt_template("evaluation_prompt")
        if not evaluation_prompt_template:
             logging.error("Evaluation prompt template not found. Cannot proceed.")
             return None, config_loader # Return loader for potential use, but no evaluator

        # Create Evaluator Instance
        evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template)
        logging.info(f"Evaluator initialized successfully with model: {evaluator_model_name} (Type: {evaluator_llm_type})")
        return evaluator, config_loader

    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        return None, None
    except KeyError as e:
        logging.error(f"Missing expected key in config file: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}", exc_info=True)
        return None, None


def process_result_file(filepath: str, evaluator: Evaluator):
    """
    Loads a result file, re-evaluates entries with eval_error=true,
    recalculates metrics, and saves the updated file if changes were made.
    """
    logging.debug(f"Processing file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Skipping invalid JSON file: {filepath}")
        return
    except Exception as e:
        logging.error(f"Skipping file due to read error: {filepath} - {e}")
        return

    needs_processing = False
    items_to_reevaluate = []

    # Check if re-evaluation is needed and identify items
    if "per_dataset_details" in data:
        for dataset_name, dataset_content in data["per_dataset_details"].items():
            if "results" in dataset_content:
                for i, result in enumerate(dataset_content["results"]):
                    if isinstance(result, dict) and result.get("eval_error") is True:
                        needs_processing = True
                        items_to_reevaluate.append(
                            {"dataset": dataset_name, "index": i, "item": result}
                        )
            else:
                logging.warning(f"Missing 'results' key in dataset '{dataset_name}' in file: {filepath}")
    else:
        logging.warning(f"Missing 'per_dataset_details' key in file: {filepath}")
        return # Cannot process without this structure

    if not needs_processing:
        logging.debug(f"No evaluation errors found in: {filepath}. Skipping re-evaluation.")
        return

    logging.info(f"Found {len(items_to_reevaluate)} items with eval_error=true in: {filepath}. Starting re-evaluation.")
    changes_made = False

    # Perform re-evaluation
    for item_info in items_to_reevaluate:
        dataset_name = item_info["dataset"]
        index = item_info["index"]
        result_item = item_info["item"]

        question = result_item.get("question")
        model_answer = result_item.get("model_answer")
        expected_answer = result_item.get("expected_answer")

        if not all([question, model_answer, expected_answer is not None]):
            logging.warning(f"  Skipping re-evaluation for item {index} in dataset '{dataset_name}' due to missing data (question/model_answer/expected_answer).")
            # Keep eval_error=True
            continue

        logging.info(f"  Re-evaluating item {index} in dataset '{dataset_name}'...")
        try:
            new_judgment = evaluator.evaluate_answer(
                question,
                model_answer,
                str(expected_answer) # Ensure expected answer is string
            )
            # Normalize judgment
            normalized_judgment = new_judgment.strip().lower() if isinstance(new_judgment, str) else "error_invalid_type"

            logging.info(f"    Original judgment: {result_item.get('self_evaluation')}, New judgment: {normalized_judgment}")

            # Update the result item in the original data structure
            data["per_dataset_details"][dataset_name]["results"][index]["self_evaluation"] = normalized_judgment
            # Mark error as resolved IF the new judgment is valid ('yes' or 'no')
            if normalized_judgment in ["yes", "no"]:
                 data["per_dataset_details"][dataset_name]["results"][index]["eval_error"] = False
                #  logging.info(f"    Evaluation successful. Marked eval_error as False.")
            else:
                 # Keep eval_error as True if the evaluator still returns an error or unexpected value
                 data["per_dataset_details"][dataset_name]["results"][index]["eval_error"] = True
                 logging.warning(f"    Evaluation resulted in unexpected judgment: '{normalized_judgment}'. Kept eval_error as True.")

            changes_made = True # Mark that we modified the data

        except Exception as e:
            logging.error(f"    Error during re-evaluation for item {index} in dataset '{dataset_name}': {e}", exc_info=True)
            # Keep eval_error=True, maybe update judgment to indicate re-eval error
            data["per_dataset_details"][dataset_name]["results"][index]["self_evaluation"] = "error_reeval_exception"
            data["per_dataset_details"][dataset_name]["results"][index]["eval_error"] = True
            changes_made = True # Still a change, even if it's an error state


    # Recalculate metrics and save if changes were made
    if changes_made:
        logging.info(f"Re-evaluation complete for {filepath}. Recalculating overall metrics...")
        all_valid_results_for_metrics = []
        for dataset_name, dataset_content in data.get("per_dataset_details", {}).items():
             for result in dataset_content.get("results", []):
                 # Collect results that are now valid for metrics calculation
                 if (isinstance(result, dict) and
                     result.get("eval_error") is False and # Must not have error
                     result.get("self_evaluation") in ["yes", "no"]): # Must have valid judgment
                     all_valid_results_for_metrics.append(result)

        logging.info(f"  Collected {len(all_valid_results_for_metrics)} results for metric recalculation.")
        try:
            new_overall_metrics = calculate_metrics(all_valid_results_for_metrics)
            data["overall_metrics"] = new_overall_metrics
            logging.info(f"  New overall metrics calculated: {new_overall_metrics}")

            # Save the updated data back to the original file
            logging.info(f"Saving updated results to: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"Successfully updated and saved: {filepath}")

        except Exception as e:
            logging.error(f"Failed to recalculate metrics or save file {filepath}: {e}", exc_info=True)
    else:
         logging.info(f"No changes were successfully made during re-evaluation for {filepath}. File not saved.")


def main(results_dir: str = DEFAULT_RESULTS_DIR, config_path: str = DEFAULT_CONFIG_PATH):
    """
    Main function to initialize components and process result files.
    """
    logging.info("--- Starting Re-evaluation Script ---")
    # Use the paths defined relative to the project root
    logging.info(f"Using results directory: {results_dir}")
    logging.info(f"Using configuration: {config_path}")

    if not os.path.isdir(results_dir):
        logging.error(f"Results directory not found: {results_dir}")
        return

    evaluator, _ = initialize_components(config_path) # We don't need config_loader after init

    if not evaluator:
        logging.error("Failed to initialize Evaluator. Exiting.")
        return

    logging.info("--- Scanning for result files with evaluation errors ---")
    processed_files = 0
    found_files = 0
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            found_files += 1
            filepath = os.path.join(results_dir, filename)
            try:
                process_result_file(filepath, evaluator)
                processed_files +=1
            except Exception as e:
                logging.error(f"An unexpected error occurred processing file {filepath}: {e}", exc_info=True)

    logging.info(f"--- Re-evaluation Script Finished ---")
    logging.info(f"Scanned {found_files} JSON files in '{results_dir}'.")
    # Note: 'processed_files' counts files attempted, not necessarily files modified.
    # A more accurate count of modified files would require return values from process_result_file.


if __name__ == "__main__":
    # You can optionally add command-line argument parsing here
    # to specify results_dir and config_path if needed.
    # Example:
    # import argparse
    # parser = argparse.ArgumentParser(description="Re-evaluate errors in RAG test results.")
    # parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory containing result JSON files.")
    # parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the configuration file.")
    # args = parser.parse_args()
    # main(results_dir=args.results_dir, config_path=args.config)

    # Run with default paths calculated relative to project root
    main()
