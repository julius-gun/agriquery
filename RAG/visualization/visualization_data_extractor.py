# visualization_data_extractor.py
import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Any
import sys # Import sys

# Add project root to path to allow importing ConfigLoader
current_script_dir = os.path.dirname(__file__)
visualization_dir = current_script_dir # Assuming this script is directly in visualization
project_root_dir = os.path.dirname(visualization_dir) # Get the parent dir (RAG)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
except ImportError:
    print("Error: Could not import ConfigLoader. Make sure it's accessible.")
    # Define a default list if ConfigLoader fails, or raise an error
    DEFAULT_LANGUAGES = ['english', 'french', 'german'] # Fallback
    ConfigLoader = None


def get_known_languages(config_path: str = 'config.json') -> List[str]:
    """Loads known languages from the config file."""
    if ConfigLoader:
        try:
            # Construct the absolute path to config.json relative to the project root
            abs_config_path = os.path.join(project_root_dir, config_path)
            if not os.path.exists(abs_config_path):
                 print(f"Warning: Config file not found at '{abs_config_path}'. Using default languages.")
                 return DEFAULT_LANGUAGES

            config_loader = ConfigLoader(abs_config_path)
            language_configs = config_loader.config.get("language_configs", [])
            languages = [lc.get("language") for lc in language_configs if lc.get("language")]
            if languages:
                print(f"Loaded known languages from config: {languages}")
                return languages
            else:
                print("Warning: No languages found in config file. Using default languages.")
                return DEFAULT_LANGUAGES
        except Exception as e:
            print(f"Error loading languages from config: {e}. Using default languages.")
            return DEFAULT_LANGUAGES
    else:
        print("Warning: ConfigLoader not available. Using default languages.")
        return DEFAULT_LANGUAGES


# Rename function and modify logic
def extract_detailed_visualization_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for result JSON files, parses filenames and content
    to extract test parameters, F1 scores, and detailed dataset success rates.

    Args:
        results_dir: The path to the directory containing the result JSON files.

    Returns:
        A pandas DataFrame containing the extracted data. Each row represents
        either an overall F1 score or a specific dataset's success rate for a run.
        Includes columns like 'metric_type' ('f1_score' or 'dataset_success')
        and 'metric_value'. For dataset success, includes 'dataset_type'.
        Returns None if the directory doesn't exist or no valid files are found.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at '{results_dir}'")
        return None

    # Get the list of known languages for validation AND for the regex
    known_languages = get_known_languages()
    if not known_languages:
        print("Error: Could not determine known languages. Aborting extraction.")
        return None

    # --- MODIFIED Regex ---
    # Dynamically build the language part of the regex
    # Escape languages in case they contain special regex characters (unlikely but safe)
    escaped_languages = [re.escape(lang) for lang in known_languages]
    language_pattern_part = f"({'|'.join(escaped_languages)})" # e.g., (english|french|german)

    # Construct the full pattern using the dynamic language part
    # Group 1: Algorithm (\w+)
    # Group 2: Language (dynamic pattern part)
    # Group 3: Model Name (.+?) - Non-greedy match up to the next part
    # Named Groups: chunk, overlap, topk (\d+)
    filename_pattern_str = rf"(\w+)_{language_pattern_part}_(.+?)_(?P<chunk>\d+)_overlap_(?P<overlap>\d+)_topk_(?P<topk>\d+)\.json"
    print(f"Using Regex: {filename_pattern_str}") # Debug print for the regex being used
    filename_pattern = re.compile(filename_pattern_str)
    # --- END MODIFICATION ---


    extracted_data = []
    skipped_files = []

    print(f"Scanning directory: {results_dir}")
    print(f"Using known languages for validation: {known_languages}") # Added print
    for filename in os.listdir(results_dir):
        match = filename_pattern.match(filename)
        if match and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            # Reset print statement for clarity based on successful match
            print(f"\nProcessing matched file: {filename}")
            try:
                # Extract parameters from filename using the regex groups
                retrieval_algorithm = match.group(1)
                language = match.group(2) # Group 2 is now the validated language
                question_model_name = match.group(3) # Group 3 is the model name
                chunk_size_str = match.group('chunk')
                overlap_size_str = match.group('overlap')
                num_retrieved_docs_str = match.group('topk')

                print(f"  Extracted Params: algo='{retrieval_algorithm}', lang='{language}', model='{question_model_name}', chunk='{chunk_size_str}', overlap='{overlap_size_str}', topk='{num_retrieved_docs_str}'")

                # Proceed only if language is valid
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  JSON loaded successfully.") # Added print

                # Convert numeric parts after validation
                chunk_size = int(chunk_size_str)
                overlap_size = int(overlap_size_str)
                num_retrieved_docs = int(num_retrieved_docs_str)

                base_record = {
                    'retrieval_algorithm': retrieval_algorithm,
                    'language': language,
                    'question_model': question_model_name,
                    'chunk_size': chunk_size,
                    'overlap_size': overlap_size,
                    'num_retrieved_docs': num_retrieved_docs,
                    'filename': filename # Add filename for easier debugging
                }

                overall_metrics = data.get('overall_metrics', {})

                # 1. Extract F1 Score (as before, but add metric type info)
                f1_score = overall_metrics.get('f1_score')
                if f1_score is not None:
                    f1_record = base_record.copy()
                    f1_record.update({
                        'metric_type': 'f1_score',
                        'metric_value': float(f1_score),
                        'dataset_type': None # Not applicable for F1
                    })
                    extracted_data.append(f1_record)
                    print(f"  Extracted F1 score: {f1_score:.4f}")
                else:
                    print(f"  Warning: 'f1_score' not found or is null in overall_metrics.")

                # 2. Extract Dataset Success Rates
                dataset_success = overall_metrics.get('dataset_self_evaluation_success', {})
                if isinstance(dataset_success, dict) and dataset_success:
                    print(f"  Found dataset success rates: {list(dataset_success.keys())}")
                    for dataset_name, success_rate in dataset_success.items():
                        if success_rate is not None:
                            dataset_record = base_record.copy()
                            dataset_record.update({
                                'metric_type': 'dataset_success',
                                'metric_value': float(success_rate),
                                'dataset_type': dataset_name # Store the dataset name
                            })
                            extracted_data.append(dataset_record)
                            print(f"    Extracted '{dataset_name}': {success_rate:.4f}")
                        else:
                            print(f"    Warning: Success rate for '{dataset_name}' is null. Skipping.")
                elif not dataset_success:
                     print(f"  Info: 'dataset_self_evaluation_success' dictionary is empty or missing.")
                else:
                     print(f"  Warning: 'dataset_self_evaluation_success' is not a dictionary or is invalid. Skipping dataset rates.")


            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                skipped_files.append(filename + " (JSON error)")
            except ValueError as e:
                 print(f"Warning: Error converting value in {filename} (e.g., int conversion): {e}. Skipping.")
                 skipped_files.append(filename + " (value error)")
            except Exception as e:
                print(f"Warning: An unexpected error occurred processing {filename}: {e}. Skipping.")
                skipped_files.append(filename + " (unexpected error)")
        elif filename.endswith(".json"): # Only report mismatch for JSON files
             # Check if it *would* have matched if not for the language part
             # Basic check: does it contain digits_overlap_digits_topk_digits.json?
             fallback_match = re.search(r"_\d+_overlap_\d+_topk_\d+\.json$", filename)
             if fallback_match:
                 print(f"Info: Skipping file {filename} (doesn't match expected pattern, potentially due to language mismatch or structure before model name).")
                 skipped_files.append(filename + " (pattern mismatch)")
             # else: # Optionally log files that don't even look like results files
             #    print(f"Debug: Skipping non-result JSON file {filename}")


    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files due to errors or pattern mismatch.")
        # Optionally print the list of skipped files for debugging:
        # print("Skipped files:", skipped_files)


    if not extracted_data:
        print("No valid result data found to create a DataFrame.")
        return None

    df = pd.DataFrame(extracted_data)
    print(f"\nSuccessfully extracted {len(extracted_data)} data points (including F1 and dataset metrics) into DataFrame.")
    return df

if __name__ == '__main__':
    # Example usage: Assuming this script is in 'visualization' and 'results' is in the parent directory ('RAG')
    default_results_dir = os.path.join(project_root_dir, 'results')

    print(f"\n--- Testing Detailed Data Extractor ---")
    print(f"Looking for results in: {default_results_dir}")
    # Call the renamed function
    df_results = extract_detailed_visualization_data(default_results_dir)

    if df_results is not None:
        print("\n--- Extracted DataFrame Head ---")
        print(df_results.head())
        print("\n--- Unique Languages Found ---")
        print(df_results['language'].unique()) # Verify only expected languages are present
        print("\n--- Unique Question Models Found ---")
        print(df_results['question_model'].unique()) # Verify model names look correct
        print("\n--- DataFrame Info ---")
        df_results.info()
        print("\n--- Value Counts for 'metric_type' ---")
        print(df_results['metric_type'].value_counts())
        print("\n--- Value Counts for 'dataset_type' (where metric_type is dataset_success) ---")
        print(df_results[df_results['metric_type'] == 'dataset_success']['dataset_type'].value_counts())
        print("\n--- Example rows for dataset_success ---")
        print(df_results[df_results['metric_type'] == 'dataset_success'].head())

    else:
        print("\nNo DataFrame generated.")
    print(f"--- Detailed Data Extractor Test Finished ---")