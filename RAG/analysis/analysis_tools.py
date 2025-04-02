import json


def analyze_evaluation_results(evaluation_metrics, dataset_type):
    """
    Analyzes evaluation metrics and provides a breakdown.
    Now accepts a dictionary of metrics calculated by metrics.py.
    """
    if not evaluation_metrics:
        print("No evaluation results provided for analysis.")
        return

    print(f"Analysis for dataset type: {dataset_type}")

    if "accuracy" in evaluation_metrics:
        accuracy = evaluation_metrics["accuracy"]
        print(f"  Accuracy: {accuracy:.2f}%")
    else:
        print(
            f"  No accuracy found in evaluation metrics for dataset type: {dataset_type}"
        )

    if "precision" in evaluation_metrics:
        precision = evaluation_metrics["precision"]
        print(f"  Precision: {precision:.2f}%")
    else:
        print(
            f"  No precision found in evaluation metrics for dataset type: {dataset_type}"
        )

    if "recall" in evaluation_metrics:
        recall = evaluation_metrics["recall"]
        print(f"  Recall: {recall:.2f}%")
    else:
        print(
            f"  No recall found in evaluation metrics for dataset type: {dataset_type}"
        )

    if "f1_score" in evaluation_metrics:
        f1_score = evaluation_metrics["f1_score"]
        print(f"  F1 Score: {f1_score:.2f}%")
    else:
        print(
            f"  No F1 Score found in evaluation metrics for dataset_type: {dataset_type}"
        )

    # if "total_questions" in evaluation_metrics:
    #     total_questions = evaluation_metrics["total_questions"]
    #     print(f"  Total Questions: {total_questions}")
    # else:
    #     print("  Total Questions: N/A")

    # if "true_positives" in evaluation_metrics:
    #     true_positives = evaluation_metrics["true_positives"]
    #     print(f"  True Positives: {true_positives}")
    # else:
    #     print("  True Positives: N/A")

    # if "true_negatives" in evaluation_metrics:
    #     true_negatives = evaluation_metrics["true_negatives"]
    #     print(f"  True Negatives: {true_negatives}")
    # else:
    #     print("  True Negatives: N/A")

    # if "false_positives" in evaluation_metrics:
    #     false_positives = evaluation_metrics["false_positives"]
    #     print(f"  False Positives: {false_positives}")
    # else:
    #     print("  False Positives: N/A")

    # if "false_negatives" in evaluation_metrics:
    #     false_negatives = evaluation_metrics["false_negatives"]
    #     print(f"  False Negatives: {false_negatives}")
    # else:
    #     print("  False Negatives: N/A")

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
