# utils/result_manager.py
import json
import os
import re
from typing import List, Dict, Optional, Any # Added Optional, Any

class ResultManager:
    """Manages loading and saving test results to JSON files with specific naming conventions."""

    def __init__(self, output_dir: str):
        """
        Initializes the ResultManager with the output directory for results.
        """
        self.output_dir = output_dir
        # Ensure the base directory exists, useful if relative paths are used later
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizes a string component for use in a filename."""
        # Remove characters invalid for Windows/Linux filenames
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
        # Replace sequences of underscores/spaces with a single underscore
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        # Remove leading/trailing underscores/spaces
        sanitized = sanitized.strip('_')
        # Handle potential empty strings after sanitization
        if not sanitized:
            return "invalid_name"
        return sanitized

    def _generate_filename(
        self,
        retrieval_algorithm: str,
        file_identifier: str,
        question_model_name: str,
        chunk_size: int,
        overlap_size: int,
        num_retrieved_docs: int
    ) -> str:
        """Generates the filename for the results file based on test parameters."""
        sanitized_model_name = self.sanitize_filename(question_model_name)
        # Format: {retrieval_algorithm}_{file_identifier}_{sanitized_question_model_name}_{chunk_size}_overlap_{overlap_size}_topk_{num_retrieved_docs}.json
        filename = (
            f"{retrieval_algorithm}_{file_identifier}_{sanitized_model_name}_"
            f"{chunk_size}_overlap_{overlap_size}_topk_{num_retrieved_docs}.json"
        )
        return filename

    def load_previous_results(
        self,
        retrieval_algorithm: str,
        file_identifier: str,
        question_model_name: str,
        chunk_size: int,
        overlap_size: int,
        num_retrieved_docs: int
    ) -> Optional[Dict[str, Any]]:
        """Loads previous results from the JSON file if it exists, using generated filename."""
        filename = self._generate_filename(
            retrieval_algorithm, file_identifier, question_model_name,
            chunk_size, overlap_size, num_retrieved_docs
        )
        filepath = os.path.join(self.output_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_results = json.load(f)
                    print(f"Loaded previous results from {filepath}")
                    return loaded_results
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filepath}. Returning None.")
                return None
            except Exception as e:
                print(f"Error loading file {filepath}: {e}. Returning None.")
                return None
        else:
            # print(f"No previous results file found at {filepath}.") # Optional: Less verbose
            return None

    def save_results(
        self,
        results: Dict[str, Any],
        retrieval_algorithm: str,
        file_identifier: str,
        question_model_name: str,
        chunk_size: int,
        overlap_size: int,
        num_retrieved_docs: int
    ):
        """Saves the results to a JSON file, using generated filename."""
        filename = self._generate_filename(
            retrieval_algorithm, file_identifier, question_model_name,
            chunk_size, overlap_size, num_retrieved_docs
        )
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Ensure the directory for the file exists (though __init__ already created the base)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results to {filepath}: {e}")
