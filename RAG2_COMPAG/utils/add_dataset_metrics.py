# utils/add_dataset_metrics.py
import os
import json
import logging
import sys
from typing import Dict, List, Optional

# --- Path Setup ---
# Ensure the project root is in sys.path to allow importing ConfigLoader
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)  # RAG directory
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
except ImportError as e:
    logging.error(
        f"Failed to import ConfigLoader: {e}. Ensure script is run correctly relative to project structure."
    )
    sys.exit(1)

# --- Constants ---
# Define paths relative to the project root
DEFAULT_RESULTS_DIR = os.path.join(project_root_dir, "results")
DEFAULT_CONFIG_PATH = os.path.join(project_root_dir, "config.json")
# The key under which the new dataset metrics will be stored within overall_metrics
DATASET_METRIC_KEY = "dataset_self_evaluation_success"

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_target_dataset_names(config_path: str) -> List[str]:
    """Loads the config and returns the list of dataset names."""
    try:
        config_loader = ConfigLoader(config_path)
        dataset_paths = config_loader.get_question_dataset_paths()
        if not dataset_paths:
            logging.warning(
                f"No 'question_dataset_paths' found in config: {config_path}"
            )
            return []
        return list(dataset_paths.keys())
    except Exception as e:
        logging.error(
            f"Error loading dataset names from config {config_path}: {e}", exc_info=True
        )
        return []


def process_result_file(filepath: str, target_dataset_names: List[str]):
    """
    Loads a result file, calculates per-dataset 'yes' ratios, adds them
    under overall_metrics, and saves the file.
    """
    logging.debug(f"Processing file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Skipping invalid JSON file: {filepath}")
        return
    except FileNotFoundError:
        logging.error(
            f"File not found during processing (should not happen if listed): {filepath}"
        )
        return
    except Exception as e:
        logging.error(f"Skipping file due to read error: {filepath} - {e}")
        return

    if "per_dataset_details" not in data:
        logging.warning(
            f"Skipping file: Missing 'per_dataset_details' key in {filepath}"
        )
        return

    dataset_metrics = {}
    changes_made = False

    for dataset_name in target_dataset_names:
        if dataset_name in data["per_dataset_details"]:
            dataset_content = data["per_dataset_details"][dataset_name]
            results_list = dataset_content.get("results", [])

            if not isinstance(results_list, list):
                logging.warning(
                    f"Dataset '{dataset_name}' in {filepath} has invalid 'results' format (not a list). Skipping dataset."
                )
                continue

            total_count = 0
            yes_count = 0
            for result_item in results_list:
                if isinstance(result_item, dict):
                    total_count += 1
                    # Check if self_evaluation is exactly "yes" (case-sensitive, as stored)
                    if result_item.get("self_evaluation") == "yes":
                        yes_count += 1
                else:
                    logging.warning(
                        f"Found non-dictionary item in results for dataset '{dataset_name}' in {filepath}. Skipping item."
                    )

            if total_count > 0:
                rate = yes_count / total_count
                dataset_metrics[dataset_name] = rate
                logging.debug(
                    f"  Dataset '{dataset_name}': {yes_count}/{total_count} = {rate:.4f}"
                )
            else:
                # If dataset exists but has 0 results, store 0.0 rate
                dataset_metrics[dataset_name] = 0.0
                logging.debug(f"  Dataset '{dataset_name}': 0 results found.")
        # else: # If dataset from config is not in this specific result file, don't add a key for it.
        #    logging.debug(f"  Dataset '{dataset_name}' not found in this result file.")

    if dataset_metrics:  # Only proceed if we calculated any metrics
        # Ensure 'overall_metrics' exists
        if "overall_metrics" not in data:
            logging.warning(
                f"'overall_metrics' key missing in {filepath}. Creating it."
            )
            data["overall_metrics"] = {}
        elif not isinstance(data["overall_metrics"], dict):
            logging.warning(
                f"'overall_metrics' in {filepath} is not a dictionary. Overwriting with new metrics."
            )
            data["overall_metrics"] = {}  # Overwrite if it's not a dict

        # Check if the key already exists and if the content is different
        existing_metrics = data["overall_metrics"].get(DATASET_METRIC_KEY)
        if existing_metrics != dataset_metrics:
            data["overall_metrics"][DATASET_METRIC_KEY] = dataset_metrics
            changes_made = True
            logging.info(f"Adding/Updating '{DATASET_METRIC_KEY}' in {filepath}")
        else:
            logging.debug(
                f"'{DATASET_METRIC_KEY}' already exists and matches calculated metrics in {filepath}. No changes needed."
            )

    if changes_made:
        try:
            # Save the updated data back to the original file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"Successfully updated and saved: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save updated file {filepath}: {e}", exc_info=True)
    else:
        logging.debug(f"No changes made to {filepath}.")


def main(
    results_dir: str = DEFAULT_RESULTS_DIR, config_path: str = DEFAULT_CONFIG_PATH
):
    """
    Main function to find result files and process them.
    """
    logging.info("--- Starting Script to Add Dataset Evaluation Success Rates ---")
    logging.info(f"Using results directory: {results_dir}")
    logging.info(f"Using configuration: {config_path}")

    if not os.path.isdir(results_dir):
        logging.error(f"Results directory not found: {results_dir}")
        return

    target_dataset_names = get_target_dataset_names(config_path)
    if not target_dataset_names:
        logging.error("Could not retrieve target dataset names from config. Exiting.")
        return

    logging.info(f"Target dataset names loaded from config: {target_dataset_names}")
    logging.info("--- Scanning for result files ---")

    processed_files_count = 0
    found_files_count = 0
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            found_files_count += 1
            filepath = os.path.join(results_dir, filename)
            try:
                process_result_file(filepath, target_dataset_names)
                processed_files_count += 1
            except Exception as e:
                # Catch unexpected errors during the processing call itself
                logging.error(
                    f"An unexpected error occurred processing file {filepath}: {e}",
                    exc_info=True,
                )

    logging.info(f"--- Script Finished ---")
    logging.info(f"Scanned {found_files_count} JSON files in '{results_dir}'.")
    logging.info(
        f"Attempted processing on {processed_files_count} files."
    )  # Note: This counts attempts, not necessarily successful modifications.


if __name__ == "__main__":
    # Add command-line argument parsing potential here if needed
    # For now, runs with defaults relative to project root
    main()
