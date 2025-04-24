import json
import os
import glob
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

# --- Configuration ---
ZERO_SHOT_RESULTS_DIR = "results/remove_folder"
OUTPUT_DIR = "results/zeroshot_comparison"
EVALUATOR_MODEL = "gemma2_9B-8k"  # As specified in the request

# Mapping from ZeroShot dataset filenames to RAG-style dataset names
DATASET_MAP = {
    "question_answers_pairs.json": "general_questions",
    "question_answers_tables.json": "table_questions",
    "question_answers_unanswerable.json": "unanswerable_questions",
}
# Define which mapped dataset names are considered 'answerable' for metric calculation
ANSWERABLE_SOURCES = ["general_questions", "table_questions"]
UNANSWERABLE_SOURCES = ["unanswerable_questions"]

# --- Metric Calculation Logic (Adapted from RAG/evaluation/metrics.py) ---

def _calculate_counts(results: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Calculates TP, TN, FP, FN based on mapped dataset names and self_evaluation.
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    for result in results:
        judgment = str(result.get("self_evaluation", "")).strip().lower()
        # Use the mapped dataset name for logic
        source = str(result.get("dataset", "")).strip().lower() # Already mapped

        if not judgment or not source:
            print(f"Warning: Skipping result due to missing 'self_evaluation' or 'dataset': {result}")
            continue

        # Check against the predefined lists of answerable/unanswerable mapped names
        is_answerable = source in ANSWERABLE_SOURCES
        is_unanswerable = source in UNANSWERABLE_SOURCES

        if is_answerable:
            if judgment == "yes":
                tp += 1
            elif judgment == "no":
                fn += 1
            else:
                print(f"Warning: Unexpected judgment '{judgment}' for answerable question: {result}")
                fn += 1 # Treat unexpected as failure to answer correctly

        elif is_unanswerable:
            if judgment == "yes":
                tn += 1 # Correctly identified as unanswerable
            elif judgment == "no":
                fp += 1 # Incorrectly provided an answer (hallucinated)
            else:
                print(f"Warning: Unexpected judgment '{judgment}' for unanswerable question: {result}")
                fp += 1 # Treat unexpected as failure to identify correctly
        else:
            # This case should not happen if mapping covers all source files
            print(f"Warning: Unknown dataset source type '{source}' after mapping for result: {result}")

    return tp, tn, fp, fn

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates classification metrics based on evaluator judgments and mapped dataset types.
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

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision_denominator = tp + fp
    precision = tp / precision_denominator if precision_denominator > 0 else 0.0
    recall_denominator = tp + fn
    recall = tp / recall_denominator if recall_denominator > 0 else 0.0
    specificity_denominator = tn + fp
    specificity = tn / specificity_denominator if specificity_denominator > 0 else 0.0
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

# --- Dataset Success Rate Calculation (Adapted from RAG/rag_tester.py) ---

def calculate_dataset_success_rates(
    mapped_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculates the 'yes' success rate for each mapped dataset."""
    dataset_metrics = defaultdict(lambda: {"yes_count": 0, "total_count": 0})
    target_dataset_names = list(set(DATASET_MAP.values())) # Get unique mapped names

    for result in mapped_results:
        dataset_name = result.get("dataset") # Already mapped name
        judgment = str(result.get("self_evaluation", "")).strip().lower()

        if dataset_name in target_dataset_names:
            dataset_metrics[dataset_name]["total_count"] += 1
            # Count only valid 'yes' evaluations
            if judgment == "yes":
                 dataset_metrics[dataset_name]["yes_count"] += 1

    success_rates = {}
    for dataset_name in target_dataset_names:
        counts = dataset_metrics[dataset_name]
        if counts["total_count"] > 0:
            rate = counts["yes_count"] / counts["total_count"]
            success_rates[dataset_name] = rate
        else:
            success_rates[dataset_name] = 0.0 # Assign 0.0 if no results for this dataset

    return success_rates

# --- Filename Parsing ---

def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parses the ZeroShot filename to extract parameters."""
    # Pattern: {language}_{model_name}_{file_extension}_{context_type}_{noise_level}_results.json
    # Model names can contain hyphens, versions, etc. Be careful with splitting.
    # Let's try regex for more robustness
    pattern = r"^(?P<language>[^_]+)_" \
              r"(?P<model_name>.+)_" \
              r"(?P<file_extension>[^_]+)_" \
              r"(?P<context_type>[^_]+)_" \
              r"(?P<noise_level>\d+)_results\.json$"
    match = re.match(pattern, filename)
    if match:
        params = match.groupdict()
        try:
            # Convert noise_level to int
            params["noise_level"] = int(params["noise_level"])
            return params
        except ValueError:
            print(f"Warning: Could not parse noise level as integer in filename: {filename}")
            return None
    else:
        print(f"Warning: Could not parse filename with expected pattern: {filename}")
        return None

# --- Main Processing Function ---

def reformat_zeroshot_file(filepath: str):
    """Loads, processes, and reformats a single ZeroShot result file."""
    print(f"Processing file: {filepath}")
    filename = os.path.basename(filepath)

    # 1. Parse filename
    params = parse_filename(filename)
    if not params:
        return # Skip file if parsing failed

    # 2. Load original data
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        if not isinstance(original_data, list):
            print(f"Warning: Expected a list in {filename}, found {type(original_data)}. Skipping.")
            return
        if not original_data:
            print(f"Warning: File {filename} is empty. Skipping.")
            return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}. Skipping.")
        return
    except Exception as e:
        print(f"Error reading file {filename}: {e}. Skipping.")
        return

    # 3. Preprocess data: Map dataset names and calculate total duration
    processed_data = []
    total_duration = 0.0
    for item in original_data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item in {filename}: {item}")
            continue

        # --- Updated dataset mapping logic ---
        original_dataset_value = item.get("dataset")
        # Add strip() to handle potential leading/trailing whitespace
        original_dataset_key = original_dataset_value.strip() if original_dataset_value else None

        mapped_dataset = DATASET_MAP.get(original_dataset_key) # Use stripped key for lookup
        if not mapped_dataset:
            # Print the key we tried to look up for better debugging
            print(f"Warning: No mapping found for dataset key '{original_dataset_key}' (from value '{original_dataset_value}') in {filename}. Skipping item.")
            continue
        item["dataset"] = mapped_dataset # Update item with mapped name

        # Sum duration
        duration = item.get("duration")
        if isinstance(duration, (int, float)):
            total_duration += duration
        else:
            # Handle missing or invalid duration - maybe default to 0?
            # print(f"Warning: Missing or invalid duration in item: {item.get('question')}. Assuming 0.")
            pass

        processed_data.append(item)

    if not processed_data:
        print(f"Warning: No processable data left in {filename} after preprocessing. Skipping.")
        return

    # 4. Calculate Metrics
    overall_metrics = calculate_metrics(processed_data)
    dataset_success = calculate_dataset_success_rates(processed_data)
    overall_metrics["dataset_self_evaluation_success"] = dataset_success

    if overall_metrics.get("accuracy") == 0.0 or overall_metrics.get("f1_score") == 0.0:
        print(f"Skipping save for {filename}: Accuracy or F1 score is zero.")
        return # Exit the function before saving

    # 5. Restructure Data
    # 5.1 Test Run Parameters
    test_run_parameters = {
        "language_tested": params["language"],
        "question_model": params["model_name"],
        "evaluator_model": EVALUATOR_MODEL,
        "retrieval_algorithm": "zeroshot",
        # Include ZeroShot specific parameters if needed, e.g., context_type, noise_level?
        # Let's add them for completeness, mirroring RAG's inclusion of its params
        "context_type": params["context_type"],
        "noise_level": params["noise_level"],
        "file_extension_tested": params["file_extension"]
        # Omit RAG-specific keys: chunk_size, overlap_size, num_retrieved_docs, chroma_collection_used
    }

    # 5.2 Timing
    timing = {
        "duration_qa_phase_seconds": total_duration,
    }

    # 5.3 Per Dataset Details
    per_dataset_details = defaultdict(lambda: {"results": []})
    for item in processed_data:
        dataset_name = item["dataset"] # Use mapped name
        cleaned_item = {
            "question": item.get("question"),
            "expected_answer": item.get("expected_answer"),
            "model_answer": item.get("model_answer"),
            "page": item.get("target_page"), # Rename target_page to page
            "dataset": dataset_name,
            "self_evaluation": item.get("self_evaluation"),
            # Omit fields not in RAG format: context_type, noise_level, model_name, file_extension, duration
            # Omit RAG-specific fields: qa_error, eval_error (not present in ZeroShot source)
        }
        # Remove keys with None values if desired (optional)
        cleaned_item = {k: v for k, v in cleaned_item.items() if v is not None}
        per_dataset_details[dataset_name]["results"].append(cleaned_item)

    # 6. Final Output Structure
    final_output = {
        "test_run_parameters": test_run_parameters,
        "overall_metrics": overall_metrics,
        "timing": timing,
        "per_dataset_details": dict(per_dataset_details) # Convert defaultdict back to dict
    }

    # 7. Save Output
    output_filename = f"zeroshot_{filename}"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Successfully reformatted and saved to: {output_filepath}")
    except Exception as e:
        print(f"Error saving output file {output_filepath}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting ZeroShot Result Reformatting ---")
    print(f"Input Directory: {ZERO_SHOT_RESULTS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # Find all relevant json files in the input directory
    search_pattern = os.path.join(ZERO_SHOT_RESULTS_DIR, "*_results.json")
    file_list = glob.glob(search_pattern)

    if not file_list:
        print("No ZeroShot result files found matching the pattern.")
    else:
        print(f"Found {len(file_list)} files to process.")
        for filepath in file_list:
            reformat_zeroshot_file(filepath)

    print("--- Reformatting Complete ---")