import json
import os
from collections import Counter

def remove_duplicate_questions(folder_path):
    """
    Removes objects with duplicate "question" values from JSON files in a folder.

    Args:
        folder_path (str): The path to the folder containing JSON files.
    """

    all_questions = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    # 1. Collect all questions from all JSON files
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list): # Assuming the json file contains a list of objects
                    for obj in data:
                        if isinstance(obj, dict) and "question" in obj:
                            all_questions.append(obj["question"])
                else:
                    print(f"Warning: {file_name} does not contain a list of objects at the top level. Skipping question collection from this file.")
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: JSON decode error in file: {file_path}")
            continue

    # 2. Identify duplicate questions
    question_counts = Counter(all_questions)
    duplicate_questions = {question for question, count in question_counts.items() if count > 1}

    if not duplicate_questions:
        print("No duplicate questions found across all JSON files.")
        return

    print(f"Found duplicate questions: {len(duplicate_questions)}")

    # 3. Process each JSON file and remove objects with duplicate questions
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        updated_data = []
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    seen_questions_in_file = set() # To keep track of questions already kept in the current file
                    for obj in data:
                        if isinstance(obj, dict) and "question" in obj:
                            question = obj["question"]
                            if question in duplicate_questions and question not in seen_questions_in_file:
                                updated_data.append(obj) # Keep the first occurrence in each file
                                seen_questions_in_file.add(question)
                            elif question not in duplicate_questions:
                                updated_data.append(obj) # Keep if not a duplicate question
                        else:
                            updated_data.append(obj) # Keep objects without 'question' key as is
                else:
                    print(f"Warning: {file_name} does not contain a list of objects at the top level. Skipping object removal for this file.")
                    continue # Skip to next file if the structure is not as expected

            # 4. Write the updated data back to the JSON file
            with open(file_path, 'w') as f:
                json.dump(updated_data, f, indent=4) # Use indent for pretty formatting

            print(f"Processed file: {file_name}, removed objects with duplicate questions.")

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: JSON decode error in file: {file_path}")
            continue

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing JSON files: ")
    if not os.path.isdir(folder_path):
        print("Error: Invalid folder path.")
    else:
        remove_duplicate_questions(folder_path)
        print("Duplicate question removal process completed.")