import json
import os

def add_file_extension_to_json_objects(folder_path):
    """
    Adds "file_extension": "txt" to all objects in JSON files within a folder.

    Args:
        folder_path (str): The path to the folder containing the JSON files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list): # Assuming the json data is a list of objects
                    for obj in data:
                        if isinstance(obj, dict): # Check if each item in list is a dict (object)
                            obj["file_extension"] = "txt"
                        else:
                            print(f"Warning: Item in JSON list is not an object (dictionary) in file: {filename}")
                elif isinstance(data, dict): # If the json data is a single object
                    data["file_extension"] = "txt"
                else:
                    print(f"Warning: JSON data is not a list or a dictionary in file: {filename}")


                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4) # Use indent for pretty formatting, optional
                print(f"Successfully added 'file_extension' to objects in: {filename}")

            except FileNotFoundError:
                print(f"Error: File not found: {filename}")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON in file: {filename}. Please ensure it's valid JSON.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing the JSON files: ")
    if not os.path.isdir(folder_path):
        print("Error: Invalid folder path.")
    else:
        add_file_extension_to_json_objects(folder_path)
        print("Processing complete.")