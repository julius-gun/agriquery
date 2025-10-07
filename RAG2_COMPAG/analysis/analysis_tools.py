import json
# Make sure calculate_metrics is imported if needed elsewhere, but not directly used in analyze_evaluation_results
# from evaluation.metrics import calculate_metrics
from typing import Dict, Any # Use Any for more flexible dictionary values
from evaluation.metrics import calculate_metrics # Local import if needed


def analyze_evaluation_results(evaluation_metrics: Dict[str, Any], dataset_type: str):
    """
    Analyzes evaluation metrics and provides a breakdown.
    Accepts a dictionary of metrics, potentially including classification metrics
    (accuracy, precision, etc.) or retrieval metrics (source_hit_rate).
    """
    if not evaluation_metrics:
        print("No evaluation results provided for analysis.")
        return

    print(f"Analysis for dataset type: {dataset_type}")

    # Check for Classification Metrics (typically floats between 0 and 1, often printed as %)
    if "accuracy" in evaluation_metrics:
        accuracy = evaluation_metrics["accuracy"] * 100 # Convert to percentage for display
        print(f"  Accuracy: {accuracy:.2f}%")
    if "precision" in evaluation_metrics:
        precision = evaluation_metrics["precision"] * 100 # Convert to percentage
        print(f"  Precision: {precision:.2f}%")
    if "recall" in evaluation_metrics:
        recall = evaluation_metrics["recall"] * 100 # Convert to percentage
        print(f"  Recall: {recall:.2f}%")
    if "specificity" in evaluation_metrics:
        specificity = evaluation_metrics["specificity"] * 100 # Convert to percentage
        print(f"  Specificity: {specificity:.2f}%")
    if "f1_score" in evaluation_metrics:
        f1_score = evaluation_metrics["f1_score"] * 100 # Convert to percentage
        print(f"  F1 Score: {f1_score:.2f}%")

    # Check for Source Hit Rate (typically already a percentage)
    if "source_hit_rate" in evaluation_metrics:
        hit_rate = evaluation_metrics["source_hit_rate"]
        # Assuming it's already a percentage from evaluate_rag_pipeline
        print(f"  Source Hit Rate: {hit_rate:.2f}%")

    # Print raw counts if available (from the new calculate_metrics)
    if "true_positives" in evaluation_metrics:
        print(f"  True Positives: {evaluation_metrics['true_positives']}")
    if "true_negatives" in evaluation_metrics:
        print(f"  True Negatives: {evaluation_metrics['true_negatives']}")
    if "false_positives" in evaluation_metrics:
        print(f"  False Positives: {evaluation_metrics['false_positives']}")
    if "false_negatives" in evaluation_metrics:
        print(f"  False Negatives: {evaluation_metrics['false_negatives']}")
    if "total_questions" in evaluation_metrics:
         print(f"  Total Questions Processed (for these metrics): {evaluation_metrics['total_questions']}")


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
    print("\nDataset Analysis Across Types:") # Added newline for better spacing
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

# This function is primarily used by rag_tester.py now, which uses the new metrics
# It might not be directly called by rag_pipeline.py anymore, but we keep it for now.
def calculate_and_analyze_metrics(dataset_results, dataset_name):
    """Calculates metrics using metrics.py and analyzes them."""
    # This function assumes dataset_results is compatible with the *new* calculate_metrics
    evaluation_metrics = calculate_metrics(dataset_results)
    analyze_evaluation_results(evaluation_metrics, dataset_name)
    return evaluation_metrics # Return metrics for further use if needed
