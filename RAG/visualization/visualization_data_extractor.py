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


def extract_visualization_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for result JSON files, parses filenames and content
    to extract test parameters and F1 scores, validating the language component.

    Args:
        results_dir: The path to the directory containing the result JSON files.

    Returns:
        A pandas DataFrame containing the extracted data (parameters and f1_score),
        or None if the directory doesn't exist or no valid files are found.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at '{results_dir}'")
        return None

    # Get the list of known languages for validation
    known_languages = get_known_languages()
    if not known_languages:
        print("Error: Could not determine known languages. Aborting extraction.")
        return None

    # Regex to capture parameters from the filename
    # Groups: 1=algo, 2=lang, 3=model, 4=chunk, 5=overlap, 6=topk
    # MODIFIED Regex: Use non-greedy match for model name anchored by surrounding structure
    filename_pattern = re.compile(
        r"(\w+)_(\w+)_(.+?)_(\d+)_overlap_(\d+)_topk_(\d+)\.json"
    )

    extracted_data = []
    skipped_files = []

    print(f"Scanning directory: {results_dir}")
    for filename in os.listdir(results_dir):
        match = filename_pattern.match(filename)
        if match and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                # Extract parameters from filename using the regex
                retrieval_algorithm = match.group(1)
                language = match.group(2)
                question_model_name = match.group(3) # Group 3 is the model name
                chunk_size_str = match.group(4)
                overlap_size_str = match.group(5)
                num_retrieved_docs_str = match.group(6)

                # --- VALIDATION STEP ---
                if language not in known_languages:
                    # print(f"Debug: Skipping file {filename}. Extracted language '{language}' is not in known list: {known_languages}")
                    skipped_files.append(filename)
                    continue # Skip this file as the language part is incorrect
                # --- END VALIDATION ---

                # Proceed only if language is valid
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert numeric parts after validation
                chunk_size = int(chunk_size_str)
                overlap_size = int(overlap_size_str)
                num_retrieved_docs = int(num_retrieved_docs_str)

                # Extract F1 score safely from JSON content
                f1_score = data.get('overall_metrics', {}).get('f1_score')

                if f1_score is not None:
                    extracted_data.append({
                        'retrieval_algorithm': retrieval_algorithm,
                        'language': language,
                        'question_model': question_model_name, # Use the correctly extracted model name
                        'chunk_size': chunk_size,
                        'overlap_size': overlap_size,
                        'num_retrieved_docs': num_retrieved_docs,
                        'f1_score': float(f1_score) # Ensure it's a float
                    })
                    # Optional: Add a print statement during debugging to verify extracted names
                    # print(f"  Extracted from {filename}: lang={language}, model={question_model_name}, f1={f1_score:.4f}")
                else:
                    print(f"Warning: 'f1_score' not found or is null in {filename}. Skipping.")
                    skipped_files.append(filename + " (missing f1_score)")


            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                skipped_files.append(filename + " (JSON error)")
            except ValueError as e:
                 print(f"Warning: Error converting value in {filename} (e.g., int conversion): {e}. Skipping.")
                 skipped_files.append(filename + " (value error)")
            except Exception as e:
                print(f"Warning: An unexpected error occurred processing {filename}: {e}. Skipping.")
                skipped_files.append(filename + " (unexpected error)")
        # else:
            # Optional: print a message for files that don't match the pattern
            # if filename.endswith(".json"): # Only print for JSON files that didn't match
            #    print(f"Info: Skipping file {filename} as it doesn't match the expected pattern.")
            #    skipped_files.append(filename + " (pattern mismatch)")


    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files due to invalid language, errors, or pattern mismatch.")
        # Optionally print the list of skipped files for debugging:
        # print("Skipped files:", skipped_files)


    if not extracted_data:
        print("No valid result data found to create a DataFrame.")
        return None

    df = pd.DataFrame(extracted_data)
    print(f"\nSuccessfully extracted data from {len(df)} valid files.")
    return df

if __name__ == '__main__':
    # Example usage: Assuming this script is in 'visualization' and 'results' is in the parent directory ('RAG')
    default_results_dir = os.path.join(project_root_dir, 'results')

    print(f"\n--- Testing Data Extractor ---")
    print(f"Looking for results in: {default_results_dir}")
    df_results = extract_visualization_data(default_results_dir)

    if df_results is not None:
        print("\n--- Extracted DataFrame Head ---")
        print(df_results.head())
        print("\n--- Unique Languages Found ---")
        print(df_results['language'].unique()) # Verify only expected languages are present
        print("\n--- Unique Question Models Found ---")
        print(df_results['question_model'].unique()) # Verify model names look correct
        print("\n--- DataFrame Info ---")
        df_results.info()
        # print("\n--- Basic Description ---")
        # print(df_results.describe())
    else:
        print("\nNo DataFrame generated.")
    print(f"--- Data Extractor Test Finished ---")