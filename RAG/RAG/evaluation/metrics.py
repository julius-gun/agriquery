from typing import List, Dict, Tuple


def calculate_accuracy(results: List[Dict]) -> Tuple[float, int, int, int, int]:
    """Calculate accuracy metrics, including true negatives."""
    total_questions = len(results)
    correct_answers = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0  # Add false positives
    false_negatives = 0  # Add false negatives

    # print(f"Results for accuracy calculation: {results}") # Debugging line
    for result in results:
        dataset = result.get("dataset", "")
        evaluation = result.get("self_evaluation", "").lower()

        if "unanswerable" in dataset:  # Handle unanswerable questions
            if "yes" in evaluation:
                true_negatives += 1
            else:
                false_positives += 1  # Model gave an answer when it shouldn't have
        else:  # Handle answerable questions
            if "yes" in evaluation:
                true_positives += 1
            else:
                false_negatives += 1  # Model failed to give the correct answer

    # Avoid division by zero
    correct_answers = true_positives + true_negatives
    accuracy = (correct_answers / total_questions) if total_questions > 0 else 0

    return (
        accuracy,
        correct_answers,
        total_questions,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    )


def calculate_precision(results: List[Dict]) -> float:
    """Calculate precision from evaluation results.
    Precision: (True Correct Claims)/(Total Generated Claims)
    """
    true_positives = 0
    false_positives = 0
    # true_negatives = 0

    for result in results:
        dataset = result.get("dataset", "")
        evaluation = result.get("self_evaluation", "").lower()
        model_answer = result.get("model_answer", "").lower()

        if "unanswerable" in dataset:
            if "yes" not in evaluation:  # Model should answer "unknown"
                false_positives += 1  # Model gave an answer when it shouldn't have
            # else:
            #     true_negatives += 1
        else:
            if "yes" in evaluation:
                true_positives += 1

    # print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}") # Debugging line
    total_claims = true_positives + false_positives
    precision = (true_positives / total_claims) if total_claims > 0 else 0
    return precision


def calculate_recall(results: List[Dict]) -> float:
    """Calculate recall from evaluation results.
    Recall: (True Correct Claims)/(Total Known Facts)
    """
    true_positives = 0
    false_negatives = 0

    for result in results:
        dataset = result.get("dataset", "")
        evaluation = result.get("self_evaluation", "").lower()

        if (
            "unanswerable" not in dataset
        ):  # Only consider answerable questions for recall
            if "yes" in evaluation:
                true_positives += 1
            else:
                false_negatives += 1

    total_known_facts = true_positives + false_negatives  # Only answerable questions
    recall = (true_positives / total_known_facts) if total_known_facts > 0 else 0
    return recall


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 Score from precision and recall."""
    if (precision + recall) == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate all metrics: accuracy, precision, recall, F1 score, TP, TN, FP, FN.
        Also includes all raw counts for more detailed analysis."""
    accuracy, _, _, true_positives, true_negatives, false_positives, false_negatives = (
        calculate_accuracy(results)
    )
    precision = calculate_precision(results)
    recall = calculate_recall(results)
    f1 = calculate_f1_score(precision, recall)
    # print(f"Calculated Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}") # Debugging line
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,  # Include these for detailed analysis
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


if __name__ == "__main__":
    # Example usage with a mix of answerable and unanswerable questions:
    example_results = [
        {
            "question": "Q1",
            "self_evaluation": "yes",
            "dataset": "pairs",
            "model_answer": "Answer",
        },
        {
            "question": "Q2",
            "self_evaluation": "yes",
            "dataset": "pairs",
            "model_answer": "Answer",
        },
        {
            "question": "Q3",
            "self_evaluation": "no",
            "dataset": "pairs",
            "model_answer": "Wrong",
        },
        {
            "question": "Q4",
            "self_evaluation": "yes",
            "dataset": "tables",
            "model_answer": "Answer",
        },
        {
            "question": "Q5",
            "self_evaluation": "no",
            "dataset": "tables",
            "model_answer": "Wrong",
        },
        {
            "question": "U1",
            "self_evaluation": "no",
            "dataset": "unanswerable",
            "model_answer": "Unknown",
        },  # Correctly unknown
        {
            "question": "U2",
            "self_evaluation": "no",
            "dataset": "unanswerable",
            "model_answer": "Some Answer",
        },  # Incorrectly answered
        {
            "question": "U3",
            "self_evaluation": "no",
            "dataset": "unanswerable",
            "model_answer": "Not found in context",
        },  # Correctly unknown
    ]

    metrics = calculate_metrics(example_results)
    print("Metrics:", metrics)
    # Expected output (approximately):
    # Metrics: {'accuracy': 75.0, 'precision': 0.75, 'recall': 1.0, 'f1_score': 0.8571428571428571, 'true_positives': 3, 'true_negatives': 2, 'false_positives': 1, 'false_negatives': 2}

    accuracy_val, correct_count, total_count, tp, tn, fp, fn = calculate_accuracy(
        example_results
    )
    print(
        f"Accuracy: {accuracy_val:.2f}%, Correct answers: {correct_count}, Total questions: {total_count}"
    )
    print(
        f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}"
    )
    precision_val = calculate_precision(example_results)
    print(f"Precision: {precision_val:.2f}")
    recall_val = calculate_recall(example_results)
    print(f"Recall: {recall_val:.2f}")
    f1_score_val = calculate_f1_score(precision_val, recall_val)
    print(f"F1 Score: {f1_score_val:.2f}")
