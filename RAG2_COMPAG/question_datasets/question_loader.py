import json
import os
from typing import List, Dict, Any

class QuestionLoader:
    """Loads questions and answers from JSON files."""

    def __init__(self, dataset_paths: List[str]):
        """
        Initializes the QuestionLoader.

        Args:
            dataset_paths (List[str]): A list of paths to question dataset JSON files.
        """
        self.dataset_paths = dataset_paths
        self.questions_data = self._load_questions_from_files()

    def _load_questions_from_files(self) -> List[Dict[str, Any]]:
        """Loads questions from all specified dataset files."""
        all_questions = []
        for path in self.dataset_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list): # Check if the loaded data is a list
                        all_questions.extend(data)
                    else:
                        raise ValueError(f"Dataset file '{path}' does not contain a list of questions.")
            except FileNotFoundError:
                print(f"Warning: Question dataset file not found at '{path}'. Skipping.")
            except json.JSONDecodeError:
                print(f"Warning: Error decoding JSON from '{path}'. Skipping.")
            except ValueError as ve:
                print(f"Warning: {ve} Skipping file '{path}'.")
            except Exception as e:
                print(f"Warning: Error loading questions from '{path}': {e}. Skipping.")
        return all_questions

    def get_questions(self) -> List[Dict[str, Any]]:
        """Returns all loaded questions."""
        return self.questions_data

if __name__ == '__main__':
    # Example usage:
    dataset_paths_example = [
        "question_answers_pairs.json",
        "question_answers_tables.json",
        "question_answers_unanswerable.json" # Assuming these files are in the same directory for example
    ]

    # Create dummy json files for testing if they don't exist
    for file_path in dataset_paths_example:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([{"question": "example question", "answer": "example answer", "page": 1}], f, indent=4)


    question_loader = QuestionLoader(dataset_paths_example)
    questions = question_loader.get_questions()

    print("Loaded questions:")
    for q in questions:
        print(q)