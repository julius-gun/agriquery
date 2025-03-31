# utils/question_loader.py
import json
from typing import List, Dict
import os
class QuestionLoader:
    """Loads questions from JSON datasets."""

    def __init__(self, dataset_paths: List[str]):
        """Initializes the QuestionLoader with a list of dataset paths."""
        self.dataset_paths = dataset_paths

    def load_questions(self) -> List[Dict]:
        """Loads questions from all configured datasets."""
        all_questions = []
        for dataset_path in self.dataset_paths:
            try:
                with open(dataset_path, "r") as f:
                    questions = json.load(f)
                    # Add dataset source information
                    for q in questions:
                        # q["dataset"] = dataset_path # filename will be used for language
                        q["dataset"] = os.path.basename(dataset_path) # filename will be used for language
                        
                    all_questions.extend(questions)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading questions from {dataset_path}: {e}")
        return all_questions