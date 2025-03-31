# python -m visualization.visualization_data_extractor
# visualization/visualization_data_extractor.py
import json
import os
import re
from typing import Dict, List, Tuple
import numpy as np

from utils.metrics import calculate_metrics

def extract_metrics_from_results(results_dir: str) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]]]:
    """
    Extracts performance metrics from JSON result files and structures the data for visualization.

    The data is structured as follows:
    {
        model_name: {
            file_extension: {
                dataset: { # Added dataset level
                    language: {
                        noise_level: {
                            "accuracy": ...,
                            "precision": ...,
                            "recall": ...,
                            "f1_score": ...,
                            "duration_mean": ...,
                        }
                    }
                }
            }
        }
    }
    """
    visualization_data = {}
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and "_results.json" in f]
    if not result_files:
        print(f"Warning: No result files (*_results.json) found in '{results_dir}'.")
        return visualization_data # Return empty dict if no files are found

    for filename in result_files:


        match = re.match(r"([a-z]+)_(.+)_([a-z]+)_([a-z]+)_(\d+)_results\.json", filename)
        if not match:
            print(f"Warning: Filename '{filename}' does not match expected pattern and will be skipped.")
            continue

        language, model_name_part, file_extension, context_type, noise_level_str = match.groups()
        model_name = model_name_part.replace("_", "-")
        noise_level = int(noise_level_str)
        filepath = os.path.join(results_dir, filename)

        try:
            with open(filepath, "r", encoding='utf-8') as f:
                results_list = json.load(f)
                if not results_list:
                    print(f"Warning: {filename} is empty or contains no results.")
                    continue

                metrics = calculate_metrics(results_list)
                durations = [result["duration"] for result in results_list]
                duration_mean = np.mean(durations) if durations else np.nan

                # Extract dataset from the first result object (assuming consistent dataset in file)
                dataset = results_list[0].get("dataset", "unknown_dataset") # Default to "unknown_dataset" if not found

                if model_name not in visualization_data:
                    visualization_data[model_name] = {}
                if file_extension not in visualization_data[model_name]:
                    visualization_data[model_name][file_extension] = {}
                if dataset not in visualization_data[model_name][file_extension]: # Add dataset level
                    visualization_data[model_name][file_extension][dataset] = {}
                if language not in visualization_data[model_name][file_extension][dataset]:
                    visualization_data[model_name][file_extension][dataset][language] = {}


                visualization_data[model_name][file_extension][dataset][language][noise_level] = {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "duration_mean": duration_mean,
                }

        except FileNotFoundError:
            print(f"Error: {filepath} not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON in {filepath}. Please ensure it is valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

    return visualization_data


def save_visualization_data(visualization_data: Dict, output_dir: str, filename: str = "visualization_data.json"):
    """Saves the processed visualization data to a JSON file."""
    os.makedirs(os.path.join(output_dir, "visualization", "data"), exist_ok=True)
    filepath = os.path.join(output_dir, "visualization", "data", filename)
    with open(filepath, 'w') as f:
        json.dump(visualization_data, f, indent=4)
    print(f"Visualization data saved to {filepath}")


if __name__ == "__main__":
    results_directory = "results"  # Replace with your actual results directory
    visualization_data = extract_metrics_from_results(results_directory)
    save_visualization_data(visualization_data, results_directory)
    print("Example of extracted visualization data structure (first model, file extension, dataset, language, noise level):")
    if visualization_data:
        first_model = next(iter(visualization_data))
        first_file_extension = next(iter(visualization_data[first_model]))
        first_dataset = next(iter(visualization_data[first_model][first_file_extension])) # first dataset
        first_language = next(iter(visualization_data[first_model][first_file_extension][first_dataset]))
        first_noise_level = next(iter(visualization_data[first_model][first_file_extension][first_dataset][first_language]))
        print(visualization_data[first_model][first_file_extension][first_dataset][first_language][first_noise_level])
    else:
        print("No visualization data extracted.")