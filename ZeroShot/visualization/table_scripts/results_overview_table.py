# visualization/table_scripts/results_overview_table.py
# python -m visualization.table_scripts.results_overview_table
import json
import os
import re
from typing import Dict, List, Tuple
import numpy as np

def extract_summary_data_from_results(results_dir: str) -> List[Dict[str, str]]:
    """
    Extracts summary data (model, extension, language, noise level, dataset) from result filenames,
    grouping rows with the same model, extension, language, and dataset and showing noise levels
    as a comma-separated string.
    Returns a list of dictionaries, sorted by model, extension, language, and dataset.
    """
    summary_data = []
    grouped_data = {} # Use a dictionary to group data by model, extension, language, dataset

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and "_results.json" in f]
    if not result_files:
        print(f"Warning: No result files (*_results.json) found in '{results_dir}'.")
        return summary_data

    for filename in result_files:
        match = re.match(r"([a-z]+)_(.+)_([a-z]+)_([a-z]+)_(\d+)_results\.json", filename)
        if match:
            language, model_name_part, file_extension, context_type, noise_level_str = match.groups()
            model_name = model_name_part.replace("_", "-")
            noise_level = int(noise_level_str)



            group_key = (model_name, file_extension, language)
            if group_key not in grouped_data:
                grouped_data[group_key] = []
            grouped_data[group_key].append(noise_level)
        else:
            print(f"Warning: Filename '{filename}' does not match expected pattern and will be skipped for summary.")

    # Convert grouped data to summary data list
    for key, noise_levels in grouped_data.items():
        model_name, file_extension, language = key
        # Store noise levels as a sorted list of integers
        sorted_noise_levels = sorted(noise_levels)

        summary_data.append({
            "model_name": model_name,
            "file_extension": file_extension,
            "language": language,
            "noise_level": sorted_noise_levels, # Store sorted list of noise levels
        })

    # Sort summary data by model_name, file_extension, language, and noise_level
    summary_data.sort(key=lambda x: (x["model_name"], x["file_extension"], x["language"], x["noise_level"]))
    return summary_data

def create_summary_table_markdown(results_dir: str) -> str:
    """
    Generates a markdown table summarizing the models, extensions, languages, noise levels, and datasets tested.

    Args:
        results_dir (str): Directory containing JSON result files.

    Returns:
        str: Markdown table as a string.
    """
    summary_data = extract_summary_data_from_results(results_dir)

    if not summary_data:
        return "No results data found to generate summary table."

    markdown_table = "| Model | Extension | Language | Noise Level |\n" # Added Dataset column
    markdown_table += "|---|---|---|---|\n"

    for item in summary_data:
        # Convert noise level list back to comma-separated string for markdown
        noise_level_str = ", ".join(map(str, item['noise_level']))
        markdown_table += f"| {item['model_name']} | {item['file_extension']} | {item['language']} | {noise_level_str} |\n"

    return markdown_table

if __name__ == "__main__":
    results_directory = "results"  # Replace with your actual results directory if needed
    markdown_table = create_summary_table_markdown(results_directory)

    artifact_content = f"""
{markdown_table}
"""

    print("Here is the summary table in markdown format:")
    print(artifact_content)

    # Artifact for markdown table
    print("\n--- Artifact ---")
    print(f""":::artifact{{identifier="results-summary-table" type="text/markdown" title="Results Summary Table"}}```markdown {artifact_content}::: """)