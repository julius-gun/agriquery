# python -m visualization.table_scripts.incorrect_questions_table
# visualization/table_scripts/incorrect_questions_table.py
import json
import os
from collections import defaultdict

def analyze_incorrect_questions(results_dir: str, token_sizes=[1000, 10000, 30000]):
    """
    Analyzes result files to find the top 10 most frequently incorrectly answered questions
    for each dataset, specifically for given token sizes, and states the total answer count.

    Args:
        results_dir (str): Directory containing JSON result files.
        token_sizes (list of int): List of token sizes to analyze.
    """

    dataset_names = {
        "question_answers_pairs": "General Questions",
        "question_answers_tables": "Table Questions",
        "question_answers_unanswerable": "Unanswerable Questions"
    }

    for token_size in token_sizes:
        incorrect_question_counts = defaultdict(lambda: defaultdict(int))
        total_question_counts = defaultdict(lambda: defaultdict(int))

        print(f"\n--- Top 10 Most Incorrectly Answered Questions ({token_size} Tokens) ---")

        for filename in os.listdir(results_dir):
            if not filename.endswith(".json") or "_results.json" not in filename or f"token_{token_size}" not in filename:
                continue

            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    results_list = json.load(f)
                    for result in results_list:
                        dataset_key = result.get("dataset").split('/')[-1].replace('.json', '')
                        question = result.get("question")

                        if dataset_key in dataset_names:
                            total_question_counts[dataset_key][question] += 1
                            if result.get("self_evaluation") == "no":
                                incorrect_question_counts[dataset_key][question] += 1
                        else:
                            total_question_counts["unknown_dataset"][question] += 1
                            if result.get("self_evaluation") == "no":
                                incorrect_question_counts["unknown_dataset"][question] += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {filename}. Skipping.")
            except FileNotFoundError:
                print(f"File {filename} not found. Skipping.")

        for dataset_key, questions_data in incorrect_question_counts.items():
            if not questions_data:
                continue

            sorted_questions = sorted(questions_data.items(), key=lambda item: item[1], reverse=True)
            top_10_questions = sorted_questions[:10]

            dataset_display_name = dataset_names.get(dataset_key, dataset_key.replace('_', ' ').title())
            print(f"\nDataset: {dataset_display_name}")
            if top_10_questions:
                for question, count in top_10_questions:
                    total_count = total_question_counts[dataset_key][question]
                    print(f"- Question: \"{question}\", Incorrect Count: {count} out of {total_count}")
            else:
                print(f"No incorrectly answered questions found for this dataset with {token_size} tokens.")


if __name__ == "__main__":
    results_directory = "results"
    analyze_incorrect_questions(results_directory)