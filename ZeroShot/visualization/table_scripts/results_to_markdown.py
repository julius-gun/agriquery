# visualization/table_scripts/results_to_markdown.py

# python -m visualization.table_scripts.results_to_markdown                 
import json
import os
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from utils.config_loader import ConfigLoader
from utils.metrics import calculate_metrics


def load_test_results(results_dir: str) -> Dict[Tuple[str, str, str], Dict[str, Dict[int, List[Dict]]]]:
    """
    Loads test results from JSON files in the specified directory.

    Results are structured as:
    {
        (model_name, file_extension, context_type): {
            language: {
                noise_level: [list of result dictionaries]
            }
        }
    }
    """
    all_results = {}

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json") or "_results.json" not in filename:
            continue

        match = re.match(r"([a-z]+)_(.+)_([a-z]+)_([a-z]+)_(\d+)_results\.json", filename)
        if not match:
            continue

        language, model_name_part, file_extension, context_type, noise_level_str = match.groups()
        model_name = model_name_part.replace("_", "-")
        noise_level = int(noise_level_str)
        filepath = os.path.join(results_dir, filename)

        try:
            with open(filepath, "r") as f:
                results_list = json.load(f)
                if not results_list:
                    print(f"Warning: {filename} is empty or contains no results.")
                    continue

                result_key = (model_name, file_extension, context_type)
                if result_key not in all_results:
                    all_results[result_key] = {}
                if language not in all_results[result_key]:
                    all_results[result_key][language] = {}
                all_results[result_key][language][noise_level] = results_list

        except FileNotFoundError:
            print(f"Error: {filepath} not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON in {filepath}. Please ensure it is valid JSON.")

    return all_results


def create_markdown_table(
    model_name: str,
    file_extension: str,
    context_type: str,
    language_data: Dict[str, Dict[int, List[Dict]]],
) -> str:
    """
    Generates a Markdown table for a given model and its test results.
    """
    markdown_output = f"## Model: {model_name}, File Extension: {file_extension}, Context Type: {context_type}\n\n"

    noise_levels = set()
    languages = sorted(language_data.keys())
    for lang in languages:
        noise_levels.update(language_data[lang].keys())
    sorted_noise_levels = sorted(list(noise_levels))

    header_row_metrics = ["Noise Level", "Mean Duration(s)", "", "", "Accuracy", "", "", "Precision", "", "", "Recall", "", "", "F1 Score", "", ""]
    header_row_languages = ["", "English", "French", "German", "English", "French", "German", "English", "French", "German", "English", "French", "German", "English", "French", "German"]
    metric_headers = ["Mean Duration(s)", "Accuracy", "Precision", "Recall", "F1 Score"]

    table_data = [header_row_metrics, header_row_languages]

    for noise_level in sorted_noise_levels:
        row = [noise_level]
        for metric_name in metric_headers:
            for lang in languages:
                results_list = language_data.get(lang, {}).get(noise_level, [])
                metric_value = calculate_metric_value(results_list, metric_name)
                row.append(metric_value)
        table_data.append(row)

    markdown_table = format_table_to_markdown(table_data, header_row_metrics, header_row_languages)
    markdown_output += markdown_table + "\n\n---\n"
    return markdown_output


def calculate_metric_value(results_list: List[Dict], metric_name: str):
    """
    Calculates a specific metric value from a list of results.
    Rounds "Mean Duration(s)" to 1 decimal place and other metrics to 3.
    """
    if not results_list:
        return "x"
        # return np.nan

    if metric_name == "Mean Duration(s)":
        durations = [result["duration"] for result in results_list]
        metric_value = np.mean(durations) if durations else np.nan
        return round(metric_value, 1) if isinstance(metric_value, (float, np.float64)) else metric_value
    else:
        metrics = calculate_metrics(results_list)
        metric_key = metric_name.lower().replace(" ", "_")
        metric_value = metrics.get(metric_key, np.nan)
        return round(metric_value, 3) if isinstance(metric_value, (float, np.float64)) else metric_value


def format_table_to_markdown(table_data: List[List[str]], header_row_metrics: List[str], header_row_languages: List[str]) -> str:
    """
    Formats the table data into a Markdown table string.
    """
    markdown_table = "| " + " | ".join(header_row_metrics) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header_row_metrics)) + " |\n"
    markdown_table += "| " + " | ".join(header_row_languages) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header_row_languages)) + " |\n"

    for row_data in table_data[2:]:  # Start from the data rows, skip header rows
        markdown_table += "| " + " | ".join(map(str, row_data)) + " |\n"
    return markdown_table


def write_markdown_report(markdown_content: str, output_file: str):
    """
    Writes the generated Markdown content to the specified output file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as md_file:
            md_file.write(markdown_content)
        print(f"\nMarkdown report generated to {output_file}")
    except Exception as e:
        print(f"Error writing markdown file: {e}")


def results_to_markdown(output_file: str = "results/all_results.md"):
    """
    Converts JSON results files to a Markdown report, including comparison tables for each model.
    """
    config_loader = ConfigLoader()
    results_dir = config_loader.get_output_dir()
    all_results = load_test_results(results_dir)

    markdown_output = "# LLM Test Results\n\n"
    for key, language_data in all_results.items():
        model_name, file_extension, context_type = key
        markdown_table = create_markdown_table(model_name, file_extension, context_type, language_data)
        markdown_output += markdown_table

    write_markdown_report(markdown_output, output_file)


if __name__ == "__main__":
    results_to_markdown(output_file="results/all_results.md")
