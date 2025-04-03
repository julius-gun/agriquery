# evaluation/metrics.py
from typing import List, Dict, Tuple, Any

def _calculate_counts(results: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Calculates True Positives (TP), True Negatives (TN),
    False Positives (FP), and False Negatives (FN) based on evaluation results.

    Logic:
    - Positive Condition: Question is answerable (source is 'general_questions' or 'table_questions').
    - Negative Condition: Question is unanswerable (source is 'unanswerable_questions').

    Mapping:
    - TP: Answerable source AND Evaluator judgment is "yes".
    - FN: Answerable source AND Evaluator judgment is "no".
    - TN: Unanswerable source AND Evaluator judgment is "yes" (QA correctly identified as unanswerable).
    - FP: Unanswerable source AND Evaluator judgment is "no" (QA hallucinated an answer).

    Args:
        results: A list of dictionaries, where each dictionary represents an
                 evaluated question and must contain:
                 - 'self_evaluation': The evaluator LLM's judgment ("yes" or "no").
                                      (Note: Renamed from 'evaluator_judgment' in request
                                      to match current usage in rag_tester.py)
                 - 'dataset': The source dataset identifier (e.g., "general_questions",
                              "table_questions", "unanswerable_questions").

    Returns:
        A tuple containing the counts: (tp, tn, fp, fn).
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    # Define answerable and unanswerable dataset identifiers based on common patterns
    # These should match the keys used in config.json's question_dataset_paths
    answerable_sources = ["general_questions", "table_questions", "pairs", "tables"]
    unanswerable_sources = ["unanswerable_questions", "unanswerable"]

    for result in results:
        # Use .get() with default empty string and .lower() for robustness
        judgment = str(result.get("self_evaluation", "")).strip().lower()
        source = str(result.get("dataset", "")).strip().lower()

        if not judgment or not source:
            print(f"Warning: Skipping result due to missing 'self_evaluation' or 'dataset': {result}")
            continue

        is_answerable = any(ans_src in source for ans_src in answerable_sources)
        is_unanswerable = any(unans_src in source for unans_src in unanswerable_sources)

        if is_answerable:
            # Positive Condition: Question is answerable
            if judgment == "yes":
                tp += 1  # Correctly answered an answerable question
            elif judgment == "no":
                fn += 1  # Failed to answer an answerable question correctly
            else:
                print(f"Warning: Unexpected judgment '{judgment}' for answerable question: {result}")
                fn += 1  # Model failed to give the correct answer

        elif is_unanswerable:
            # Negative Condition: Question is unanswerable
            if judgment == "yes":
                tn += 1  # Correctly identified as unanswerable (QA likely said "Unknown", Eval agreed)
            elif judgment == "no":
                fp += 1  # Incorrectly provided an answer (QA hallucinated, Eval disagreed)
            else:
                print(f"Warning: Unexpected judgment '{judgment}' for unanswerable question: {result}")
                fp += 1  # Model failed to give the correct answer
        else:
            print(f"Warning: Unknown dataset source type '{source}' for result: {result}")

    return tp, tn, fp, fn


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates classification metrics (Accuracy, Precision, Recall, Specificity, F1-Score)
    based on evaluator judgments and source dataset types.

    Args:
        results: A list of dictionaries, where each dictionary represents an
                 evaluated question and must contain:
                 - 'self_evaluation': The evaluator LLM's judgment ("yes" or "no").
                 - 'dataset': The source dataset identifier (e.g., "general_questions",
                              "table_questions", "unanswerable_questions").

    Returns:
        A dictionary containing the calculated metrics (as floats between 0.0 and 1.0)
        and the raw counts (as integers). Returns 0.0 for metrics with zero denominators.
        Example:
        {
            "accuracy": 0.85,
            "precision": 0.9,
            "recall": 0.8,
            "specificity": 0.7,
            "f1_score": 0.847,
            "true_positives": 80,
            "true_negatives": 70,
            "false_positives": 10,
            "false_negatives": 20,
            "total_questions": 180
        }
    """
    if not results:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "specificity": 0.0, "f1_score": 0.0,
            "true_positives": 0, "true_negatives": 0,
            "false_positives": 0, "false_negatives": 0,
            "total_questions": 0
        }

    tp, tn, fp, fn = _calculate_counts(results)
    total = tp + tn + fp + fn

    # --- Calculate Metrics ---

    # Accuracy: (TP + TN) / Total
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Precision: TP / (TP + FP)
    precision_denominator = tp + fp
    precision = tp / precision_denominator if precision_denominator > 0 else 0.0

    # Recall (Sensitivity): TP / (TP + FN)
    recall_denominator = tp + fn
    recall = tp / recall_denominator if recall_denominator > 0 else 0.0

    # Specificity: TN / (TN + FP)
    specificity_denominator = tn + fp
    specificity = tn / specificity_denominator if specificity_denominator > 0 else 0.0

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_denominator = precision + recall
    f1_score = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_questions": total
    }


if __name__ == "__main__":
    # Example usage with a mix of answerable and unanswerable questions:
    # Using dataset names consistent with config.json
    example_results = [
        # Answerable, Correctly Answered (TP)
        {"question": "Q1", "self_evaluation": "yes", "dataset": "general_questions", "model_answer": "Correct"}, # TP 1
        {"question": "Q2", "self_evaluation": "yes", "dataset": "general_questions", "model_answer": "Correct"}, # TP 2
        {"question": "Q3", "self_evaluation": "yes", "dataset": "table_questions",   "model_answer": "Correct"}, # TP 3
        # Answerable, Incorrectly Answered (FN)
        {"question": "Q4", "self_evaluation": "no", "dataset": "general_questions", "model_answer": "Wrong"},   # FN 1
        {"question": "Q5", "self_evaluation": "no", "dataset": "table_questions",   "model_answer": "Wrong"},   # FN 2
        # Unanswerable, Correctly Identified (TN) - Evaluator says 'yes' because QA answer matches expected 'Unknown'
        {"question": "U1", "self_evaluation": "yes", "dataset": "unanswerable_questions", "model_answer": "Unknown"}, # TN 1
        {"question": "U2", "self_evaluation": "yes", "dataset": "unanswerable_questions", "model_answer": "Not found"}, # TN 2
        # Unanswerable, Incorrectly Answered (FP) - Evaluator says 'no' because QA answer doesn't match expected 'Unknown'
        {"question": "U3", "self_evaluation": "no", "dataset": "unanswerable_questions", "model_answer": "Some Answer"}, # FP 1
        # Example with slightly different casing/spacing
        {"question": "Q6", "self_evaluation": " YES ", "dataset": " general_questions ", "model_answer": "Correct"}, # TP 4
        {"question": "U4", "self_evaluation": " NO ", "dataset": " unanswerable_questions ", "model_answer": "Another Answer"}, # FP 2
        # Example with missing data (should be skipped with warning)
        {"question": "Q7", "dataset": "general_questions", "model_answer": "Correct"},
        {"question": "Q8", "self_evaluation": "yes", "model_answer": "Correct"},
        # Example with unknown dataset type (should be skipped with warning)
        {"question": "Q9", "self_evaluation": "yes", "dataset": "future_dataset", "model_answer": "Correct"},
    ]

    # Expected Counts from valid entries (10 total valid):
    # TP = 4 (Q1, Q2, Q3, Q6)
    # FN = 2 (Q4, Q5)
    # TN = 2 (U1, U2)
    # FP = 2 (U3, U4)
    # Total = 10

    # Expected Metrics:
    # Accuracy = (TP + TN) / Total = (4 + 2) / 10 = 6 / 10 = 0.6
    # Precision = TP / (TP + FP) = 4 / (4 + 2) = 4 / 6 = 0.666...
    # Recall = TP / (TP + FN) = 4 / (4 + 2) = 4 / 6 = 0.666...
    # Specificity = TN / (TN + FP) = 2 / (2 + 2) = 2 / 4 = 0.5
    # F1 = 2 * (Prec * Rec) / (Prec + Rec) = 2 * (0.666 * 0.666) / (0.666 + 0.666) = 0.666...

    print("--- Calculating Metrics from Example Results ---")
    metrics = calculate_metrics(example_results)

    print("\nCalculated Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\n--- Expected Metrics ---")
    print("  Accuracy: 0.6000")
    print("  Precision: 0.6667")
    print("  Recall: 0.6667")
    print("  Specificity: 0.5000")
    print("  F1 Score: 0.6667")
    print("  True Positives: 4")
    print("  True Negatives: 2")
    print("  False Positives: 2")
    print("  False Negatives: 2")
    print("  Total Questions: 10")

    # Test with empty list
    print("\n--- Calculating Metrics from Empty List ---")
    empty_metrics = calculate_metrics([])
    print(empty_metrics)