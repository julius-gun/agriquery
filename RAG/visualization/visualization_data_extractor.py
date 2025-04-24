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
    Scans a directory for RAG and ZeroShot result JSON files, parses filenames
    and content to extract test parameters, F1 scores, and detailed dataset
    success rates into a unified DataFrame.

    Args:
        results_dir: The path to the directory containing the result JSON files.

    Returns:
        A pandas DataFrame containing the extracted data. Each row represents
        either an overall F1 score or a specific dataset's success rate for a run.
        Includes columns for parameters from both RAG and ZeroShot formats
        (using None/NaN where a parameter doesn't apply), 'metric_type',
        'metric_value', and 'dataset_type'.
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

    escaped_languages = [re.escape(lang) for lang in known_languages]
    # IMPORTANT: Remove the outer parentheses from language_pattern_part itself,
    # as the named group syntax (?P<name>...) provides the grouping.
    language_choices_part = '|'.join(escaped_languages) # e.g., english|french|german

    # --- Define Corrected Regex Patterns ---
    # RAG Pattern
    # Added ?P<lang> around the language choices part
    rag_pattern_str = rf"^(?P<algo>\w+)_(?P<lang>{language_choices_part})_(?P<model>.+?)_(?P<chunk>\d+)_overlap_(?P<overlap>\d+)_topk_(?P<topk>\d+)\.json$"
    rag_pattern = re.compile(rag_pattern_str)
    print(f"Using RAG Regex: {rag_pattern_str}") # Will now show (?P<lang>english|french|german)

    # ZeroShot Pattern
    # Added ?P<lang> around the language choices part
    zeroshot_pattern_str = rf"^zeroshot_(?P<lang>{language_choices_part})_(?P<model>.+?)_(?P<ext>[a-zA-Z0-9]+)_(?P<context>\w+)_(?P<noise>\d+)_results\.json$"
    zeroshot_pattern = re.compile(zeroshot_pattern_str)
    print(f"Using ZeroShot Regex: {zeroshot_pattern_str}") # Will now show (?P<lang>english|french|german)
    # --- End Regex Definitions ---


    extracted_data = []
    skipped_files = []

    print(f"\nScanning directory: {results_dir}")
    print(f"Using known languages for validation: {known_languages}")

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue # Quickly skip non-JSON files

        filepath = os.path.join(results_dir, filename)
        params = None
        file_type = None # To track which pattern matched

        # --- Try Matching Patterns ---
        rag_match = rag_pattern.match(filename)
        zeroshot_match = None # Initialize zeroshot_match
        if rag_match:
            file_type = "RAG"
            print(f"\nProcessing RAG file: {filename}")
            try:
                params = {
                    'retrieval_algorithm': rag_match.group('algo'),
                    'language': rag_match.group('lang'),
                    'question_model': rag_match.group('model'),
                    'chunk_size': int(rag_match.group('chunk')),
                    'overlap_size': int(rag_match.group('overlap')),
                    'num_retrieved_docs': int(rag_match.group('topk')),
                    # ZeroShot specific params set to None
                    'file_extension': None,
                    'context_type': None,
                    'noise_level': None,
                    'filename': filename,
                    'file_type': file_type # Add file type identifier
                }
                print(f"  Extracted RAG Params: { {k: v for k, v in params.items() if k not in ['filename', 'file_type']} }")
            except ValueError:
                print(f"  Warning: Error parsing numeric values from RAG filename {filename}. Skipping.")
                skipped_files.append(filename + " (RAG value error)")
                continue
            except IndexError as e: # Catch potential group access errors during development/debugging
                 print(f"  Error accessing regex group for RAG file {filename}: {e}. This shouldn't happen with corrected regex. Skipping.")
                 skipped_files.append(filename + " (RAG regex group error)")
                 continue
        else:
            zeroshot_match = zeroshot_pattern.match(filename)
            if zeroshot_match:
                file_type = "ZeroShot"
                print(f"\nProcessing ZeroShot file: {filename}")
                try:
                    params = {
                        'retrieval_algorithm': 'zeroshot', # Hardcoded for this pattern
                        'language': zeroshot_match.group('lang'),
                        'question_model': zeroshot_match.group('model'),
                        'file_extension': zeroshot_match.group('ext'),
                        'context_type': zeroshot_match.group('context'),
                        'noise_level': int(zeroshot_match.group('noise')),
                        # RAG specific params set to None
                        'chunk_size': None,
                        'overlap_size': None,
                        'num_retrieved_docs': None,
                        'filename': filename,
                        'file_type': file_type # Add file type identifier
                    }
                    print(f"  Extracted ZeroShot Params: { {k: v for k, v in params.items() if k not in ['filename', 'file_type']} }")
                except ValueError:
                    print(f"  Warning: Error parsing numeric values from ZeroShot filename {filename}. Skipping.")
                    skipped_files.append(filename + " (ZeroShot value error)")
                    continue
                except IndexError as e: # Catch potential group access errors
                    print(f"  Error accessing regex group for ZeroShot file {filename}: {e}. This shouldn't happen with corrected regex. Skipping.")
                    skipped_files.append(filename + " (ZeroShot regex group error)")
                    continue

        # --- Process if a pattern matched ---
        if params and file_type: # Check if params were successfully extracted
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  JSON loaded successfully.")

                # --- Extract Metrics (Common Logic) ---
                overall_metrics = data.get('overall_metrics', {})
                if not overall_metrics:
                     print(f"  Warning: 'overall_metrics' key missing or empty in {filename}. Cannot extract metrics.")
                     # Decide if you want to skip the file entirely or just metrics
                     # continue # Option: skip file if no metrics

                # 1. Extract F1 Score
                f1_score = overall_metrics.get('f1_score')
                if f1_score is not None:
                    f1_record = params.copy() # Start with parameters from filename
                    f1_record.update({
                        'metric_type': 'f1_score',
                        'metric_value': float(f1_score),
                        'dataset_type': None # Not applicable for overall F1
                    })
                    extracted_data.append(f1_record)
                    print(f"  Extracted F1 score: {f1_score:.4f}")
                elif overall_metrics: # Only warn if overall_metrics existed
                    print(f"  Warning: 'f1_score' not found or is null in overall_metrics.")

                # 2. Extract Dataset Success Rates
                dataset_success = overall_metrics.get('dataset_self_evaluation_success', {})
                if isinstance(dataset_success, dict) and dataset_success:
                    print(f"  Found dataset success rates: {list(dataset_success.keys())}")
                    for dataset_name, success_rate in dataset_success.items():
                        if success_rate is not None:
                            dataset_record = params.copy() # Start with parameters from filename
                            dataset_record.update({
                                'metric_type': 'dataset_success',
                                'metric_value': float(success_rate),
                                'dataset_type': dataset_name # Store the dataset name
                            })
                            extracted_data.append(dataset_record)
                            print(f"    Extracted '{dataset_name}': {success_rate:.4f}")
                        else:
                            print(f"    Warning: Success rate for '{dataset_name}' is null. Skipping this dataset.")
                elif overall_metrics and not dataset_success: # Only info if overall_metrics existed
                     print(f"  Info: 'dataset_self_evaluation_success' dictionary is empty or missing.")
                elif overall_metrics and not isinstance(dataset_success, dict):
                     print(f"  Warning: 'dataset_self_evaluation_success' is not a dictionary. Skipping dataset rates.")
                # No message if overall_metrics itself was missing

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                skipped_files.append(filename + " (JSON error)")
            except ValueError as e:
                 # Catch potential float conversion errors for metrics
                 print(f"Warning: Error converting metric value in {filename} to float: {e}. Skipping metric/file.")
                 skipped_files.append(filename + " (metric value error)")
            except Exception as e:
                print(f"Warning: An unexpected error occurred processing {filename}: {e}. Skipping.")
                skipped_files.append(filename + f" (unexpected error: {type(e).__name__})")

        elif filename.endswith(".json"): # If it's JSON but didn't match known patterns
             # Avoid printing for every non-matching file, only those ending in .json
             # that didn't match either RAG or ZeroShot patterns.
             if not rag_match and not zeroshot_match: # Explicitly check if both failed
                 print(f"Info: Skipping file {filename} (doesn't match known RAG or ZeroShot patterns).")
                 skipped_files.append(filename + " (pattern mismatch)")
             # else: # Optionally log files that don't even look like results files
             #    print(f"Debug: Skipping non-result JSON file {filename}")


    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files due to errors or pattern mismatch.")
        # Optionally print the list for debugging:
        for skipped in skipped_files: print(f"  - {skipped}")


    if not extracted_data:
        print("No valid result data found to create a DataFrame.")
        return None

    # --- Create DataFrame ---
    df = pd.DataFrame(extracted_data)

    # --- Optional: Define column order for consistency ---
    # Gather all unique parameter keys plus metric keys
    all_param_keys = [
        'filename', 'file_type', # Add file_type if you want it in the df
        'retrieval_algorithm', 'language', 'question_model',
        'chunk_size', 'overlap_size', 'num_retrieved_docs', # RAG specific
        'file_extension', 'context_type', 'noise_level', # ZeroShot specific
    ]
    metric_keys = ['metric_type', 'metric_value', 'dataset_type']
    # Ensure all columns exist, adding missing ones with None/NaN
    for col in all_param_keys + metric_keys:
        if col not in df.columns:
            df[col] = None # Add missing columns

    # Reorder columns (optional but good practice)
    # Put identifying params first, then specific params, then metrics
    desired_order = [
        'filename', #'file_type', # Uncomment if added above
        'retrieval_algorithm', 'language', 'question_model',
        'chunk_size', 'overlap_size', 'num_retrieved_docs',
        'file_extension', 'context_type', 'noise_level',
        'metric_type', 'metric_value', 'dataset_type'
    ]
    # Filter desired_order to only include columns actually present in the df
    final_columns = [col for col in desired_order if col in df.columns]
    df = df[final_columns]


    print(f"\nSuccessfully extracted {len(extracted_data)} data points into DataFrame with {len(df.columns)} columns.")
    print(f"DataFrame columns: {df.columns.tolist()}")
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