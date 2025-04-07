# visualization_data_extractor.py
import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Any

def extract_visualization_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for result JSON files, parses filenames and content
    to extract test parameters and F1 scores.

    Args:
        results_dir: The path to the directory containing the result JSON files.

    Returns:
        A pandas DataFrame containing the extracted data (parameters and f1_score),
        or None if the directory doesn't exist or no valid files are found.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at '{results_dir}'")
        return None

    # Regex to capture parameters from the filename
    # Groups: 1=algo, 2=lang, 3=model, 4=chunk, 5=overlap, 6=topk
    # MODIFIED Regex: Use non-greedy match for model name anchored by surrounding structure
    filename_pattern = re.compile(
        r"(\w+)_(\w+)_(.+?)_(\d+)_overlap_(\d+)_topk_(\d+)\.json"
    )

    extracted_data = []

    print(f"Scanning directory: {results_dir}")
    for filename in os.listdir(results_dir):
        match = filename_pattern.match(filename)
        if match and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract parameters from filename using the new regex
                retrieval_algorithm = match.group(1)
                language = match.group(2)
                question_model_name = match.group(3) # Group 3 is now the non-greedy model name
                chunk_size = int(match.group(4))
                overlap_size = int(match.group(5))
                num_retrieved_docs = int(match.group(6))

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
                    # print(f"  Extracted model name from {filename}: {question_model_name}")
                else:
                    print(f"Warning: 'f1_score' not found or is null in {filename}. Skipping.")

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
            except ValueError as e:
                 print(f"Warning: Error converting value in {filename} (e.g., int conversion): {e}. Skipping.")
            except Exception as e:
                print(f"Warning: An unexpected error occurred processing {filename}: {e}. Skipping.")
        # else:
            # Optional: print a message for files that don't match the pattern
            # if filename.endswith(".json"): # Only print for JSON files that didn't match
            #    print(f"Info: Skipping file {filename} as it doesn't match the expected pattern.")


    if not extracted_data:
        print("No valid result data found to create a DataFrame.")
        return None

    df = pd.DataFrame(extracted_data)
    print(f"Successfully extracted data from {len(df)} files.")
    return df

if __name__ == '__main__':
    # Example usage: Assuming this script is in 'visualization' and 'results' is in the parent directory ('RAG')
    current_script_dir = os.path.dirname(__file__)
    project_root_dir = os.path.dirname(current_script_dir) # Get the parent dir (RAG)
    default_results_dir = os.path.join(project_root_dir, 'results')

    print(f"--- Testing Data Extractor ---")
    print(f"Looking for results in: {default_results_dir}")
    df_results = extract_visualization_data(default_results_dir)

    if df_results is not None:
        print("\n--- Extracted DataFrame Head ---")
        print(df_results.head())
        print("\n--- Unique Question Models Found ---")
        print(df_results['question_model'].unique()) # Print unique models found
        print("\n--- DataFrame Info ---")
        df_results.info()
        print("\n--- Basic Description ---")
        print(df_results.describe())
    else:
        print("\nNo DataFrame generated.")
    print(f"--- Data Extractor Test Finished ---")