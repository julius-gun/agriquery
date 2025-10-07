# scripts/check_result_integrity.py
import json
import os
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Adjust path to import ConfigLoader from the parent directory's utils
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)  # Assumes RAG is the parent of scripts
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
except ImportError:
    print(
        "Error: Could not import ConfigLoader. "
        "Ensure the script is in a 'scripts' directory and 'utils' directory is at the same level as 'scripts' "
        "within the project root (e.g., RAG/scripts, RAG/utils)."
    )
    sys.exit(1)

# --- Configuration Keys (to avoid magic strings) ---
KEY_TEST_RUN_PARAMS = "test_run_parameters"
KEY_RETRIEVAL_ALGORITHM = "retrieval_algorithm"
KEY_CHUNK_SIZE = "chunk_size"
KEY_OVERLAP_SIZE = "overlap_size"
KEY_NUM_RETRIEVED_DOCS = "num_retrieved_docs"

KEY_OVERALL_METRICS = "overall_metrics"
KEY_TOTAL_QUESTIONS = "total_questions"

KEY_QA_ERROR = "qa_error"
KEY_EVAL_ERROR = "eval_error"

# Assume detailed results are under this key, and it's a list of dicts
KEY_DETAILED_RESULTS = "results_per_question"
KEY_MODEL_ANSWER = "model_answer"


def calculate_expected_total_questions(config_loader: ConfigLoader, project_root: str) -> int:
    """
    Calculates the total number of questions from all dataset paths specified in the config.
    """
    dataset_paths_dict = config_loader.get_question_dataset_paths()
    if not dataset_paths_dict:
        print("Warning: 'question_dataset_paths' not found or empty in config. Cannot calculate expected total questions.")
        return 0

    total_questions = 0
    for dataset_name, relative_path in dataset_paths_dict.items():
        if not relative_path or not isinstance(relative_path, str):
            print(f"Warning: Invalid path for dataset '{dataset_name}': {relative_path}. Skipping.")
            continue

        absolute_path = os.path.join(project_root, relative_path)
        try:
            with open(absolute_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            if isinstance(dataset, list):
                total_questions += len(dataset)
            else:
                print(f"Warning: Dataset file '{absolute_path}' for '{dataset_name}' is not a list. Cannot count questions.")
        except FileNotFoundError:
            print(f"Warning: Dataset file not found at '{absolute_path}' for '{dataset_name}'. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from '{absolute_path}' for '{dataset_name}'. Skipping.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while reading '{absolute_path}': {e}. Skipping.")
    return total_questions


def get_config_derived_expectations(config_loader: ConfigLoader) -> Tuple[Optional[int], Optional[int], Optional[int], List[str]]:
    """
    Extracts expected RAG parameters (chunk_size, overlap_size, num_retrieved_docs) 
    and the list of RAG algorithms to test from config.
    Uses the first element from *_to_test lists for chunk/overlap.
    """
    rag_params = config_loader.get_rag_parameters()
    if not rag_params:
        print("Warning: 'rag_parameters' not found in config.")
        return None, None, None, []

    expected_chunk_size = None
    chunk_sizes_to_test = rag_params.get("chunk_sizes_to_test", [])
    if chunk_sizes_to_test and isinstance(chunk_sizes_to_test, list) and len(chunk_sizes_to_test) > 0:
        expected_chunk_size = chunk_sizes_to_test[0]
    else:
        print("Warning: 'chunk_sizes_to_test' is missing, empty, or not a list in rag_parameters.")

    expected_overlap_size = None
    overlap_sizes_to_test = rag_params.get("overlap_sizes_to_test", [])
    if overlap_sizes_to_test and isinstance(overlap_sizes_to_test, list) and len(overlap_sizes_to_test) > 0:
        expected_overlap_size = overlap_sizes_to_test[0]
    else:
        print("Warning: 'overlap_sizes_to_test' is missing, empty, or not a list in rag_parameters.")

    expected_num_docs = rag_params.get("num_retrieved_docs")
    if expected_num_docs is None: # Can be 0, so check for None
        print("Warning: 'num_retrieved_docs' is missing in rag_parameters.")

    # Get the list of RAG algorithms that these parameters apply to
    rag_algorithms_to_check = config_loader.get_retrieval_algorithms_to_test() # Uses the method from ConfigLoader

    return expected_chunk_size, expected_overlap_size, expected_num_docs, rag_algorithms_to_check


def validate_single_result_file(
    file_path: str,
    expected_chunk_size: Optional[int],
    expected_overlap_size: Optional[int],
    expected_num_docs: Optional[int],
    expected_total_questions: int,
    rag_algorithms_subject_to_specific_checks: List[str]
) -> List[str]:
    """
    Validates a single result JSON file against the specified criteria.
    Returns a list of issue descriptions.
    """
    issues = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return [f"File not found at path: {file_path}"]
    except json.JSONDecodeError:
        return [f"Invalid JSON in file: {file_path}"]
    except Exception as e:
        return [f"Error reading file {file_path}: {e}"]

    test_params = data.get(KEY_TEST_RUN_PARAMS, {})
    actual_retrieval_algorithm = None

    if not test_params:
        issues.append(f"Missing '{KEY_TEST_RUN_PARAMS}' section.")
    else:
        actual_retrieval_algorithm = test_params.get(KEY_RETRIEVAL_ALGORITHM)
        if not actual_retrieval_algorithm:
            issues.append(f"Missing '{KEY_RETRIEVAL_ALGORITHM}' in '{KEY_TEST_RUN_PARAMS}'.")
        
        # 1. Conditionally check RAG-specific parameters
        # Only apply these if the file's algorithm is one that should have these params
        if actual_retrieval_algorithm and actual_retrieval_algorithm in rag_algorithms_subject_to_specific_checks:
            if expected_chunk_size is not None:
                actual_chunk_size = test_params.get(KEY_CHUNK_SIZE)
                if actual_chunk_size != expected_chunk_size:
                    issues.append(
                        f"Incorrect {KEY_CHUNK_SIZE}: Expected {expected_chunk_size}, Found {actual_chunk_size} (for RAG algo: {actual_retrieval_algorithm})."
                    )
            if expected_overlap_size is not None:
                actual_overlap_size = test_params.get(KEY_OVERLAP_SIZE)
                if actual_overlap_size != expected_overlap_size:
                    issues.append(
                        f"Incorrect {KEY_OVERLAP_SIZE}: Expected {expected_overlap_size}, Found {actual_overlap_size} (for RAG algo: {actual_retrieval_algorithm})."
                    )
            if expected_num_docs is not None:
                actual_num_docs = test_params.get(KEY_NUM_RETRIEVED_DOCS)
                if actual_num_docs != expected_num_docs:
                    issues.append(
                        f"Incorrect {KEY_NUM_RETRIEVED_DOCS}: Expected {expected_num_docs}, Found {actual_num_docs} (for RAG algo: {actual_retrieval_algorithm})."
                    )

    # 2. Check total questions (applies to all files)
    overall_metrics = data.get(KEY_OVERALL_METRICS, {})
    if not overall_metrics:
        issues.append(f"Missing '{KEY_OVERALL_METRICS}' section.")
    else:
        if expected_total_questions > 0: # Only check if we have a valid expected number
            actual_total_questions = overall_metrics.get(KEY_TOTAL_QUESTIONS)
            if actual_total_questions is None:
                 issues.append(f"Missing '{KEY_TOTAL_QUESTIONS}' in '{KEY_OVERALL_METRICS}'.")
            elif actual_total_questions != expected_total_questions:
                issues.append(
                    f"Incorrect {KEY_TOTAL_QUESTIONS}: Expected {expected_total_questions}, Found {actual_total_questions}."
                )

    # 3. Check for qa_error flag (applies to all files)
    if data.get(KEY_QA_ERROR) is True:
        issues.append(f"Flag '{KEY_QA_ERROR}: true' exists.")

    # 4. Check for eval_error flag (applies to all files)
    if data.get(KEY_EVAL_ERROR) is True:
        issues.append(f"Flag '{KEY_EVAL_ERROR}: true' exists.")

    # 5. Check for "Error" in model_answer (applies to all files, case-insensitive)
    detailed_results = data.get(KEY_DETAILED_RESULTS, [])
    if isinstance(detailed_results, list):
        for i, item in enumerate(detailed_results):
            if isinstance(item, dict):
                model_answer = item.get(KEY_MODEL_ANSWER)
                if model_answer and isinstance(model_answer, str):
                    if "error" in model_answer.lower(): # Case-insensitive check
                        issues.append(
                            f"'{KEY_MODEL_ANSWER}' (item {i+1}) contains 'Error': \"{model_answer[:50]}...\""
                        )
            else:
                issues.append(f"Item {i+1} in '{KEY_DETAILED_RESULTS}' is not a dictionary.")
    elif KEY_DETAILED_RESULTS in data :
        issues.append(f"'{KEY_DETAILED_RESULTS}' is present but not a list.")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Check integrity of RAG and other result files.")
    parser.add_argument(
        "--config",
        default=os.path.join(project_root_dir, "config.json"),
        help="Path to the configuration JSON file.",
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to the results directory. If not provided, uses 'output_dir' from config.",
    )
    args = parser.parse_args()

    print(f"Using project root: {project_root_dir}")
    print(f"Loading configuration from: {args.config}")

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    try:
        config_loader = ConfigLoader(config_path=args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    print("Configuration loaded successfully.")

    (
        expected_chunk_size,
        expected_overlap_size,
        expected_num_docs,
        rag_algorithms_to_check # Get the list of RAG algos
    ) = get_config_derived_expectations(config_loader)
    
    print(f"Expected RAG params (for algos: {rag_algorithms_to_check}): Chunk Size={expected_chunk_size}, Overlap Size={expected_overlap_size}, Num Docs={expected_num_docs}")

    expected_total_questions = calculate_expected_total_questions(config_loader, project_root_dir)
    if expected_total_questions > 0:
        print(f"Dynamically calculated expected total questions: {expected_total_questions}")
    else:
        print("Warning: Could not determine expected total questions from datasets. Check 'question_dataset_paths' in config and dataset files.")

    results_directory_path = args.results_dir
    if not results_directory_path:
        results_directory_path = config_loader.get_output_dir()
        if not os.path.isabs(results_directory_path):
            results_directory_path = os.path.join(project_root_dir, results_directory_path)
    
    print(f"Scanning result files in: {results_directory_path}")

    if not os.path.isdir(results_directory_path):
        print(f"Error: Results directory not found at '{results_directory_path}'")
        sys.exit(1)

    problematic_files_count = 0
    total_files_scanned = 0

    for filename in os.listdir(results_directory_path):
        if filename.endswith(".json"):
            total_files_scanned += 1
            file_path = os.path.join(results_directory_path, filename)
            issues = validate_single_result_file(
                file_path,
                expected_chunk_size,
                expected_overlap_size,
                expected_num_docs,
                expected_total_questions,
                rag_algorithms_to_check # Pass the list of RAG algos
            )

            if issues:
                problematic_files_count += 1
                print(f"\n--- Issues found in: {file_path} ---")
                for issue in issues:
                    print(f"  - {issue}")

    print("\n--- Integrity Check Summary ---")
    print(f"Total JSON files scanned: {total_files_scanned}")
    if problematic_files_count > 0:
        print(f"Number of files with issues: {problematic_files_count}")
    else:
        print("All scanned files conform to the specified checks.")
    print("--- Check Finished ---")

if __name__ == "__main__":
    main()