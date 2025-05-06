# Run from ZeroShot/clean_results.py
import json
import os
import glob
import sys
from typing import List, Dict, Set

# Ensure utils directory is in the Python path
# Adjust this if your script is located elsewhere relative to utils
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, "utils")
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

try:
    from config_loader import ConfigLoader
    from question_loader import QuestionLoader
except ImportError:
    print(
        "Error: Could not import ConfigLoader or QuestionLoader."
        " Make sure clean_results.py is in the ZeroShot directory"
        " or adjust the sys.path modification."
    )
    sys.exit(1)


def load_valid_questions(config_loader: ConfigLoader) -> Set[str]:
    """Loads all valid question strings from the datasets specified in the config."""
    print("Loading valid questions from source datasets...")
    dataset_paths = config_loader.get_question_dataset_paths()
    question_loader = QuestionLoader(dataset_paths)
    all_questions_data = question_loader.load_questions()
    valid_questions = {q_data["question"] for q_data in all_questions_data}
    print(f"Loaded {len(valid_questions)} unique valid questions.")
    return valid_questions


def clean_result_file(
    filepath: str, valid_questions: Set[str]
) -> bool:
    """
    Cleans a single result file by removing entries with invalid questions.

    Args:
        filepath: Path to the result JSON file.
        valid_questions: A set of valid question strings.

    Returns:
        True if the file was modified, False otherwise.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Handle potentially empty files
            content = f.read()
            if not content:
                # print(f"Skipping empty file: {filepath}")
                return False
            results_data = json.loads(content)

        if not isinstance(results_data, list):
            print(
                f"Warning: Expected a list in {filepath}, found {type(results_data)}. Skipping."
            )
            return False

        original_count = len(results_data)
        cleaned_results = [
            result
            for result in results_data
            if isinstance(result, dict) and result.get("question") in valid_questions
        ]
        cleaned_count = len(cleaned_results)

        if original_count != cleaned_count:
            print(
                f"Cleaning {os.path.basename(filepath)}: "
                f"Removed {original_count - cleaned_count} entries."
            )
            # Overwrite the file with cleaned results
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(cleaned_results, f, indent=4, ensure_ascii=False)
            return True
        # else:
            # print(f"No changes needed for {os.path.basename(filepath)}.")

    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}. Skipping.")
        return False
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {filepath}. Skipping.")
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}. Skipping.")
        return False
    return False


def main(config_path: str = "config.json"):
    """
    Main function to orchestrate the cleaning process.
    """
    print("Starting result file cleanup...")
    config_loader = ConfigLoader(config_path)
    output_dir = config_loader.get_output_dir()

    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        sys.exit(1)

    valid_questions = load_valid_questions(config_loader)

    if not valid_questions:
        print("Error: No valid questions loaded. Aborting cleanup.")
        sys.exit(1)

    result_files = glob.glob(os.path.join(output_dir, "*.json"))
    print(f"Found {len(result_files)} JSON files in '{output_dir}'.")

    modified_files_count = 0
    for filepath in result_files:
        if clean_result_file(filepath, valid_questions):
            modified_files_count += 1

    print("-" * 30)
    print(f"Cleanup complete. {modified_files_count} files were modified.")
    print("-" * 30)


if __name__ == "__main__":
    # Assuming config.json is in the same directory or accessible path
    main()