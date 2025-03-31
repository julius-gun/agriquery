import os
import json

def get_all_questions_from_datasets(dataset_paths):
    """
    Loads questions from all specified dataset files and returns a set of questions.
    """
    all_questions = set()
    for dataset_path in dataset_paths:
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if "question" in item:
                        all_questions.add(item["question"])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading questions from {dataset_path}: {e}")
    return all_questions

def clean_results_files(results_dir, dataset_paths):
    """
    Removes objects from result JSON files if their 'question' is not found in the dataset questions,
    removes duplicate questions that are in the dataset, and prints the remaining number of questions in each file.
    """
    dataset_questions = get_all_questions_from_datasets(dataset_paths)
    if not dataset_questions:
        print("No questions loaded from datasets. Exiting.")
        return

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r+', encoding='utf-8') as f:
                    results_data = json.load(f)
                    if not isinstance(results_data, list):
                        print(f"Skipping {filename} as it does not contain a list.")
                        continue

                    filtered_results = []
                    removed_count_invalid_question = 0
                    removed_count_duplicate = 0
                    seen_questions = set()

                    for result_item in results_data:
                        if "question" in result_item:
                            question = result_item["question"]
                            if question in dataset_questions: # Check if question is in dataset
                                if question not in seen_questions: # Check for duplicates
                                    filtered_results.append(result_item)
                                    seen_questions.add(question)
                                else:
                                    removed_count_duplicate += 1 # Count duplicate questions that are in dataset
                            else:
                                removed_count_invalid_question += 1 # Count questions not in dataset
                        else:
                            removed_count_invalid_question += 1 # Count items without question field

                    if removed_count_invalid_question > 0:
                        print(f"Removed {removed_count_invalid_question} objects with invalid questions (not in dataset or missing 'question' field) from {filename}.")
                    if removed_count_duplicate > 0:
                        print(f"Removed {removed_count_duplicate} duplicate questions (that are in dataset) from {filename}.")

                    f.seek(0)  # Go to the beginning of the file
                    json.dump(filtered_results, f, indent=4)
                    f.truncate() # Remove remaining part of old data if new data is shorter

                    remaining_questions_count = len(filtered_results)
                    print(f"Remaining unique questions in {filename}: {remaining_questions_count}")


            except json.JSONDecodeError:
                print(f"Error decoding JSON in {filename}. Skipping file.")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    print("Cleaning process completed.")

if __name__ == "__main__":
    results_directory = "results"  # Replace with your results directory if different
    question_dataset_files = [
        "question_datasets/question_answers_pairs.json",
        "question_datasets/question_answers_tables.json",
        "question_datasets/question_answers_unanswerable.json"
    ]

    clean_results_files(results_directory, question_dataset_files)