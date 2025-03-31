# python -m utils.results_integrity_checker
import json
import os
from typing import Dict, List

def load_questions_from_datasets(question_datasets_dir: str) -> List[Dict]:
    """
    Loads questions from all JSON files in the question_datasets directory.

    Returns:
        List[Dict]: A list of dictionaries, each containing a question.
    """
    all_questions = []
    question_files = [f for f in os.listdir(question_datasets_dir) if f.endswith(".json")]
    for file in question_files:
        filepath = os.path.join(question_datasets_dir, file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                questions = json.load(f)
                if isinstance(questions, list):
                    all_questions.extend(questions)
                else:
                    print(f"Warning: {file} does not contain a list of questions.")
        except FileNotFoundError:
            print(f"Error: {filepath} not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON in {filepath}. Please ensure it is valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file}: {e}")
    return all_questions


def check_results_integrity(results_dir: str, question_datasets_dir: str) -> None:
    """
    Checks the integrity of result files in the given directory.
    For each result file, it verifies if all questions from the datasets have been answered
    and checks for duplicate questions.

    Args:
        results_dir (str): Directory containing result files.
        question_datasets_dir (str): Directory containing question dataset files.
    """
    print("\nStarting results integrity check...")
    all_questions_data = load_questions_from_datasets(question_datasets_dir)
    all_questions_texts = {q['question'] for q in all_questions_data} # Use a set for faster lookups
    expected_question_count = len(all_questions_texts)
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and "_results.json" in f]

    if not result_files:
        print(f"Warning: No result files (*_results.json) found in '{results_dir}' for integrity check.")
        return

    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f: # Added encoding='utf-8' here
                results_list = json.load(f)
                if not results_list:
                    print(f"Warning: {filename} is empty.")
                    continue

                actual_question_count = len(results_list)
                if actual_question_count != expected_question_count:
                    print(f"Integrity Issue in {filename}:")
                    print(f"  Expected {expected_question_count} questions, but found {actual_question_count}.")

                found_questions = set()
                duplicate_questions = []
                missing_questions = set(all_questions_texts) # Initialize with all questions

                for result in results_list:
                    question = result.get('question')
                    if question in found_questions:
                        duplicate_questions.append(question)
                    else:
                        found_questions.add(question)
                        missing_questions.discard(question) # Remove found question from missing set

                if duplicate_questions:
                    print(f"Integrity Issue in {filename}:")
                    print(f"  Duplicate questions found: {duplicate_questions}")

                if missing_questions:
                    print(f"Integrity Issue in {filename}:")
                    print(f"  Missing questions: {missing_questions}")

                # if not duplicate_questions and not missing_questions and actual_question_count == expected_question_count:
                #     print(f"Integrity Check Passed for {filename}:")
                #     print(f"  - Contains {actual_question_count} results, no duplicates, and all questions from datasets are present.")


        except FileNotFoundError:
            print(f"Error: {filepath} not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON in {filepath}. Please ensure it is valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

    print("Results integrity check completed.")

if __name__ == "__main__":
    results_directory = "results"  # Example path, adjust as needed
    question_datasets_directory = "question_datasets" # Example path, adjust as needed
    check_results_integrity(results_directory, question_datasets_directory)
