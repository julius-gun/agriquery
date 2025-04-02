import json
from evaluation.metrics import calculate_metrics
from typing import Dict


def analyze_evaluation_results(evaluation_metrics, dataset_type):
    """
    Analyzes evaluation metrics and provides a breakdown.
    Now accepts a dictionary of metrics calculated by metrics.py.
    """
    if not evaluation_metrics:
        print("No evaluation results provided for analysis.")
        return

    print(f"Analysis for dataset type: {dataset_type}")


    if isinstance(evaluation_metrics, Dict): # Check if evaluation_metrics is a dictionary
        if "accuracy" in evaluation_metrics:
            accuracy = evaluation_metrics["accuracy"]
            print(f"  Accuracy: {accuracy:.2f}%")
        if "precision" in evaluation_metrics:
            precision = evaluation_metrics["precision"]
            print(f"  Precision: {precision:.2f}%")
        if "recall" in evaluation_metrics:
            recall = evaluation_metrics["recall"]
            print(f"  Recall: {recall:.2f}%")
        if "f1_score" in evaluation_metrics:
            f1_score = evaluation_metrics["f1_score"]
            print(f"  F1 Score: {f1_score:.2f}%")

    # Placeholder for future detailed analysis
    # print("\nDetailed Breakdown (Future Enhancement):")
    # print("  - Performance by Document Format: ...")
    # print("  - Performance by Language: ...")
    # print("  - Performance by Question Type: ...")


def load_dataset(dataset_path):
    """
    Loads a question-answer dataset from a JSON file.
    Assumes each line in the JSON file is a valid JSON object representing a question-answer pair.
    """
    dataset = []
    try:
        with open(
            dataset_path, "r", encoding="utf-8"
        ) as f:  # Explicit encoding for broader compatibility
            dataset = json.load(f)  # Load the entire JSON file
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in dataset file at {dataset_path}")
        return None
    return dataset


def analyze_dataset_across_types(dataset_paths):
    """
    Loads and analyzes datasets from multiple paths.
    Provides a summary of dataset types and their paths.
    """
    print("Dataset Analysis Across Types:")
    for (
        dataset_name,
        dataset_path,
    ) in dataset_paths.items():  # Iterate through dataset_paths dictionary
        dataset = load_dataset(dataset_path)
        if dataset:
            print(
                f"  Dataset Type: {dataset_name}"
            )  # Use dataset_name from dictionary key
            print(f"    Path: {dataset_path}")
            print(f"    Number of questions: {len(dataset)}")  # Basic dataset info
        else:
            print(
                f"  Dataset Type: {dataset_name}: Failed to load from {dataset_path}"
            )  # Use dataset_name here as well
def calculate_and_analyze_metrics(dataset_results, dataset_name):
    """Calculates metrics using metrics.py and analyzes them."""
    evaluation_metrics = calculate_metrics(dataset_results)
    analyze_evaluation_results(evaluation_metrics, dataset_name)
    return evaluation_metrics # Return metrics for further use if needed
