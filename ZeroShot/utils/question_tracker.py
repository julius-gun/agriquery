import json
import os
import glob
from typing import Dict

class QuestionTracker:
    """Tracks asked questions to avoid execution."""

    def __init__(self, tracker_file_path: str):
        """
        Initializes the QuestionTracker.  The tracker_file_path is kept for compatibility
        but is no longer used for storing data.
        """
        self.tracker_file_path = tracker_file_path  # Kept for compatibility, but not used
        print(f"QuestionTracker initialized. (Tracker file is not used for storage).")


    def is_question_asked(self, question_text: str, context_type: str, noise_level: int, model_name: str, file_extension: str) -> bool:
        """
        Checks if a question with the given context parameters has already been asked
        by searching through existing result files.

        Args:
            question_text (str): The text of the question.
            context_type (str): 'page' or 'token'.
            noise_level (int): The noise level (number of pages or tokens).
            model_name (str): The name of the LLM model used.
            file_extension (str): The file extension.

        Returns:
            bool: True if the question has been asked, False otherwise.
        """
        results_dir = "results"  # Assuming results are stored in the "results" directory
        if not os.path.isdir(results_dir):
            return False  # No results directory, so question hasn't been asked

        json_files = glob.glob(os.path.join(results_dir, "*.json"))

        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    results_data = json.load(f)

                if isinstance(results_data, list):
                    for result_item in results_data:
                        if self._is_matching_result(result_item, question_text, context_type, noise_level, model_name, file_extension):
                            # print(f"Question found in {file_path}")
                            return True  # Question found in this result file

            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error reading or decoding {file_path}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        return False  # Question not found in any result file


    def _is_matching_result(self, result_item: Dict, question_text: str, context_type: str, noise_level: int, model_name: str, file_extension: str) -> bool:
        """Helper function to check if a result item matches the current question parameters."""
        return (
            isinstance(result_item, dict) and
            result_item.get("question") == question_text and
            result_item.get("context_type") == context_type and
            result_item.get("noise_level") == noise_level and
            result_item.get("model_name") == model_name and
            result_item.get("file_extension") == file_extension
        )


    # def mark_question_asked(self, question_text: str, context_type: str, noise_level: int, model_name: str, file_extension: str):
    #     """
    #     This method is no longer used, as saving results implicitly marks a question as asked.
    #     It is kept (empty) for compatibility with existing code.
    #     """
    #     pass


if __name__ == '__main__':
    pass # Remove example usage