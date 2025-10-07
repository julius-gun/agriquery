# utils/recalculate_metrics.py
import json
import os
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional

# Adjust path to import from parent directory (RAG)
# This assumes the script is run from the 'utils' directory or the project root has been added to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Try importing directly first (if running from RAG root and utils is in path)
    try:
        from config_loader import ConfigLoader # If run from utils, this might work directly
    except ImportError:
        from utils.config_loader import ConfigLoader # Standard import if RAG is in path
    from evaluation.metrics import calculate_metrics
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical(f"Failed to import necessary modules. Ensure the script is run from the 'utils' directory or PYTHONPATH is set correctly.")
    logging.critical(f"Error: {e}")
    sys.exit(1) # Exit if essential imports fail

# --- Constants ---
CONFIG_PATH = "config.json" # Relative path from utils/ directory to RAG/config.json
DATASET_METRIC_KEY = "dataset_self_evaluation_success" # Key for per-dataset success metrics

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def calculate_new_timing(per_dataset_details: Dict[str, Any], target_dataset_names: List[str]) -> Dict[str, float]:
    """
    Recalculates timing by summing durations from target datasets.

    Args:
        per_dataset_details: The dictionary containing details for each dataset run.
        target_dataset_names: List of dataset names to include in the calculation.

    Returns:
        A dictionary with recalculated timing values.
    """
    total_qa_duration = 0.0
    total_eval_duration = 0.0

    for dataset_name, details in per_dataset_details.items():
        # Check if this dataset should be included based on the current config
        if dataset_name in target_dataset_names:
            # Use .get() for safety, defaulting to 0 if keys are missing
            total_qa_duration += details.get("duration_qa_seconds", 0.0)
            total_eval_duration += details.get("duration_eval_seconds", 0.0)
        # Else: Ignore datasets not in the target list (like 'table_questions' if excluded)

    overall_duration = total_qa_duration + total_eval_duration

    return {
        "overall_duration_seconds": overall_duration,
        "duration_qa_phase_seconds": total_qa_duration,
        "duration_eval_phase_seconds": total_eval_duration,
    }

def calculate_new_dataset_success_rates(per_dataset_details: Dict[str, Any], target_dataset_names: List[str]) -> Dict[str, float]:
    """
    Calculates the 'yes' success rate for each target dataset based on its results.
    Mirrors the logic from RagTester._calculate_dataset_success_rates.

    Args:
        per_dataset_details: The dictionary containing details for each dataset run.
        target_dataset_names: List of dataset names to include in the calculation.

    Returns:
        A dictionary containing the success rate for each *target* dataset found.
    """
    dataset_metrics = {}
    logging.debug("Recalculating dataset self-evaluation success rates...")

    for dataset_name in target_dataset_names:
        # Only calculate for datasets currently specified in config
        if dataset_name in per_dataset_details:
            dataset_content = per_dataset_details[dataset_name]
            results_list = dataset_content.get("results", [])

            if not isinstance(results_list, list):
                logging.warning(f"Dataset '{dataset_name}' in result file has invalid 'results' format. Skipping rate calculation.")
                dataset_metrics[dataset_name] = 0.0 # Assign default if format is wrong
                continue

            total_count = 0
            yes_count = 0
            for result_item in results_list:
                if isinstance(result_item, dict):
                    # Check if evaluation was valid ('yes'/'no') and no errors occurred
                    # This assumes the result item structure is consistent
                    evaluation = result_item.get("self_evaluation")
                    eval_error = result_item.get("eval_error", False) # Default to False if key missing

                    # Count only results where evaluation happened without error
                    # This means 'self_evaluation' is present and 'eval_error' is False
                    # We count both 'yes' and 'no' for the denominator.
                    if evaluation is not None and not eval_error:
                        total_count += 1
                        if evaluation == "yes":
                             yes_count += 1
                    # We ignore results with qa_error or eval_error for rate calculation

            if total_count > 0:
                rate = yes_count / total_count
                dataset_metrics[dataset_name] = rate
                logging.debug(f"  Dataset '{dataset_name}' success rate: {yes_count}/{total_count} = {rate:.4f}")
            else:
                # If dataset exists but has 0 valid/non-errored results, store 0.0 rate
                dataset_metrics[dataset_name] = 0.0
                logging.debug(f"  Dataset '{dataset_name}': 0 valid results found, rate = 0.0")
        # else: # Dataset from config not present in this results file
        #    logging.warning(f"  Target dataset '{dataset_name}' not found in current result file's per_dataset_details.")
        #    dataset_metrics[dataset_name] = 0.0 # Assign 0 if target dataset is missing entirely

    return dataset_metrics


def recalculate_and_update_file(filepath: str, target_dataset_names: List[str]) -> bool:
    """
    Loads a result file, recalculates metrics and timing based on target datasets,
    and overwrites the file.

    Args:
        filepath: Path to the JSON result file.
        target_dataset_names: List of dataset names to include in the calculation.

    Returns:
        True if successful, False otherwise.
    """
    try:
        logging.info(f"Processing file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- Validate Structure ---
        if "per_dataset_details" not in data:
            logging.warning(f"Skipping file: Missing 'per_dataset_details' key in {filepath}")
            return False
        if not isinstance(data["per_dataset_details"], dict):
             logging.warning(f"Skipping file: 'per_dataset_details' is not a dictionary in {filepath}")
             return False

        per_dataset_details = data["per_dataset_details"]

        # --- Filter Results for Overall Metrics ---
        # Create a flat list containing only results from target datasets
        # These results are used by calculate_metrics which expects 'dataset' and 'self_evaluation' keys
        filtered_results_flat_list = []
        for dataset_name, details in per_dataset_details.items():
            # Check if this dataset should be included based on the current config
            if dataset_name in target_dataset_names:
                 results_list = details.get("results", [])
                 if isinstance(results_list, list):
                     # Add dataset name back to each result item if needed by calculate_metrics
                     # (Assuming calculate_metrics needs 'dataset' key within each item)
                     for item in results_list:
                         if isinstance(item, dict):
                             # Ensure the 'dataset' key reflects the source dataset name
                             item['dataset'] = dataset_name
                             # Add the item to the list for overall metric calculation
                             filtered_results_flat_list.append(item)
                 else:
                      logging.warning(f"  Dataset '{dataset_name}' in {filepath} has invalid 'results' format (expected list).")
            # Else: Ignore datasets not in the target list (like 'table_questions' if excluded)

        logging.info(f"  Recalculating overall metrics based on {len(filtered_results_flat_list)} results from datasets: {target_dataset_names}")

        # --- Recalculate Overall Metrics ---
        # Pass the flat list of results from ONLY the target datasets
        new_overall_metrics = calculate_metrics(filtered_results_flat_list)

        # --- Recalculate Per-Dataset Success Rates ---
        # This uses the original per_dataset_details but filters by target_dataset_names internally
        new_dataset_success_rates = calculate_new_dataset_success_rates(per_dataset_details, target_dataset_names)
        # Add/overwrite the success rates within the overall metrics dictionary
        new_overall_metrics[DATASET_METRIC_KEY] = new_dataset_success_rates

        # --- Recalculate Timing ---
        # This also uses the original per_dataset_details but filters by target_dataset_names internally
        new_timing = calculate_new_timing(per_dataset_details, target_dataset_names)

        # --- Update Data ---
        data['overall_metrics'] = new_overall_metrics
        data['timing'] = new_timing

        # --- Overwrite File ---
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False) # <--- Added ensure_ascii=False
        logging.info(f"  Successfully updated and saved {filepath} (with UTF-8 characters preserved)")
        return True

    except FileNotFoundError:
        logging.error(f"Error: File not found at {filepath}")
        return False
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {filepath}. File might be corrupted.")
        return False
    except KeyError as e:
        logging.error(f"Error: Missing expected key {e} in {filepath}.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {filepath}: {e}", exc_info=True)
        return False


def main():
    """
    Main function to load config, find result files, and trigger recalculation.
    """
    logging.info("--- Starting Metric Recalculation Script ---")

    # --- Load Configuration ---
    try:
        # Add logging for path calculation
        logging.info(f"Script location (__file__): {__file__}")
        script_dir = os.path.dirname(__file__)
        logging.info(f"Script directory: {script_dir}")
        logging.info(f"Calculated project root (relative to script dir): {project_root}")
        config_abs_path = os.path.abspath(os.path.join(script_dir, CONFIG_PATH))
        logging.info(f"Attempting to load config from absolute path: {config_abs_path}")

        # Config path is relative to this script's location in utils/
        config_loader = ConfigLoader(config_path=config_abs_path) # Use absolute path for loading

        results_dir_relative = config_loader.get_output_dir()
        logging.info(f"Output directory read from config: '{results_dir_relative}'")

        # Results dir path is relative to project root (calculated above)
        results_dir = os.path.join(project_root, results_dir_relative) # Construct absolute path
        logging.info(f"Calculated absolute results directory path to check: {results_dir}")

        # Get dataset names *currently* defined in the config
        target_dataset_names = list(config_loader.get_question_dataset_paths().keys())

        if not target_dataset_names:
            logging.error("Error: No 'question_dataset_paths' found in config. Cannot determine target datasets.")
            return
        logging.info(f"Target datasets for recalculation (from config): {target_dataset_names}")

        if not os.path.isdir(results_dir):
             # Log the error before returning
             logging.error(f"Error: Results directory '{results_dir}' not found or is not a directory.")
             return
        logging.info(f"Confirmed results directory exists. Scanning for result files in: {results_dir}")

    except FileNotFoundError as e:
        logging.critical(f"Failed to load configuration file at '{config_abs_path}': {e}", exc_info=True)
        return
    except Exception as e:
        logging.critical(f"Failed during configuration loading or path setup: {e}", exc_info=True)
        return

    # --- Process Files ---
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    try:
        for filename in os.listdir(results_dir):
            if filename.lower().endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                if os.path.isfile(filepath): # Ensure it's a file
                    if recalculate_and_update_file(filepath, target_dataset_names):
                        processed_count += 1
                    else:
                        # Increment failed count, assuming the function logs the specific error
                        failed_count += 1
                else:
                    logging.debug(f"Skipping non-file entry: {filepath}")
                    skipped_count += 1 # It's a directory or other non-file ending in .json?
            else:
                # Skip files that don't end with .json
                logging.debug(f"Skipping non-JSON file: {filename}")
                skipped_count += 1
    except FileNotFoundError:
        # This could happen if results_dir exists initially but is deleted before os.listdir runs
        logging.error(f"Error: Results directory '{results_dir}' disappeared during processing.")
        failed_count = -1 # Indicate a major issue
    except Exception as e:
        logging.error(f"An unexpected error occurred during file processing loop: {e}", exc_info=True)
        failed_count = -1 # Indicate a major issue


    logging.info("--- Recalculation Complete ---")
    logging.info(f"Successfully processed and updated: {processed_count} files")
    if failed_count > 0:
        logging.warning(f"Failed to process/update: {failed_count} files (check logs for details)")
    elif failed_count == -1:
         logging.error("Processing loop failed unexpectedly.")
    if skipped_count > 0:
        logging.info(f"Skipped non-JSON or non-file entries: {skipped_count}")


if __name__ == "__main__":
    main()