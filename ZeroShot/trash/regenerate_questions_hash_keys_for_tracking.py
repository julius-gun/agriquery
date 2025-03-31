import json
import os
import glob

def generate_question_key(result_item):
    """
    Generates a question key from a result item, mimicking _generate_question_key in QuestionTracker.
    """
    model_name = result_item.get("model_name")
    file_extension = result_item.get("file_extension")
    context_type = result_item.get("context_type")
    noise_level = result_item.get("noise_level")
    question_text = result_item.get("question")

    if not all([model_name, file_extension, context_type, noise_level, question_text]):
        print(f"Warning: Incomplete data in result item, skipping key generation: {result_item}")
        return None

    noise_level_str = str(noise_level) # Ensure noise_level is a string for key generation
    question_hash = hash(question_text)
    key = f"{model_name}-{file_extension}-{context_type}-{noise_level_str}-{question_hash}"
    return key

def load_existing_tracker_data(tracker_file_path):
    """
    Loads existing tracked questions from the JSON file.
    Returns an empty dict if the file doesn't exist or is invalid.
    """
    if os.path.exists(tracker_file_path):
        try:
            with open(tracker_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Question tracker file '{tracker_file_path}' corrupted or empty. Starting with empty tracker.")
            return {}
    else:
        print(f"Tracker file {tracker_file_path} does not exist. Starting with empty tracker.")
        return {}

def save_tracker_data(tracker_file_path, tracker_data):
    """
    Saves the tracked questions to the JSON file.
    """
    os.makedirs(os.path.dirname(tracker_file_path), exist_ok=True) # Ensure directory exists
    with open(tracker_file_path, 'w') as f:
        json.dump(tracker_data, f, indent=4)
    print(f"Saved tracked questions to {tracker_file_path}: {len(tracker_data)}")

def populate_question_tracker_from_results(results_dir, tracker_file_path):
    """
    Populates the question tracker data from existing result JSON files.
    """
    tracked_questions = load_existing_tracker_data(tracker_file_path)
    questions_added_count = 0

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(json_files)} result JSON files in '{results_dir}'.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                results_data = json.load(f)

            if isinstance(results_data, list):
                for result_item in results_data:
                    if isinstance(result_item, dict):
                        question_key = generate_question_key(result_item)
                        if question_key:
                            if question_key not in tracked_questions:
                                tracked_questions[question_key] = True
                                questions_added_count += 1
                                # print(f"  Added question key: {question_key}") # Optional detailed logging
                            # else:
                            #     print(f"  Question key already in tracker: {question_key}") # Optional logging for already tracked questions
                    else:
                        print(f"Warning: Item in JSON list is not a dictionary in file: {file_path}")
            else:
                print(f"Warning: JSON data is not a list in file: {file_path}")

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}. Skipping file.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

    save_tracker_data(tracker_file_path, tracked_questions)
    print(f"Successfully added {questions_added_count} questions to '{tracker_file_path}'.")


if __name__ == "__main__":
    results_directory = "results"  # Adjust if your results directory is different
    tracker_file = "utils/question_tracker_data.json" # Adjust if your tracker file path is different

    if not os.path.isdir(results_directory):
        print(f"Error: Results directory '{results_directory}' not found.")
    elif not os.path.isdir(os.path.dirname(tracker_file)):
        print(f"Error: Directory for tracker file '{os.path.dirname(tracker_file)}' not found.")
    else:
        populate_question_tracker_from_results(results_directory, tracker_file)
        print("Question tracker population complete.")