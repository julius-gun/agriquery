# utils/result_manager.py
import json
import os
import re
from typing import List, Dict

class ResultManager:
    """Manages loading and saving test results to JSON files."""

    def __init__(self, output_dir: str):
        """Initializes the ResultManager with the output directory for results."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitizes the filename by removing invalid characters."""
        return re.sub(r'[\\/*?:"<>|]', "_", filename)

    def load_previous_results(
        self, language: str, model_name: str, file_extension: str, context_type: str, noise_level: int
    ) -> List[Dict]:
        """Loads previous results from the JSON file if it exists."""
        filename = self._generate_filename(language, model_name, file_extension, context_type, noise_level)
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, "r", encoding='utf-8') as f:
                    loaded_results = json.load(f)
                    # print(f"Loaded {len(loaded_results)} previous results from {filepath}")
                    return loaded_results
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filepath}. Starting fresh.")
                return []
            except UnicodeDecodeError as e:
                print(f"Encoding error reading {filepath}: {e}. The file might not be UTF-8 encoded. Starting fresh.")
                # Optionally, you could try other encodings here, but it's usually better
                # to ensure files are consistently written in UTF-8 (see next step).
                return []
        return []

    def save_results(self, results: List[Dict], language: str, model_name: str, file_extension: str, context_type: str, noise_level: int):
        """Saves the results to a JSON file."""
        filename = self._generate_filename(language, model_name, file_extension, context_type, noise_level)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filepath}") # No need to print every save

    def _generate_filename(self, language: str, model_name: str, file_extension: str, context_type: str, noise_level: int) -> str:
        """Generates the filename for the results file based on test parameters."""
        return f"{language}_{self.sanitize_filename(model_name)}_{file_extension}_{context_type}_{noise_level}_results.json"