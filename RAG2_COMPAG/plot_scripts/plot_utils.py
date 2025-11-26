import os
import sys
import json
from typing import List, Dict, Optional

# --- Path Setup ---
def add_project_paths() -> str:
    """Adds the project root directory (RAG) to sys.path and returns it."""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root should be the 'RAG' directory (parent of 'visualization')
    visualization_dir = os.path.dirname(current_script_dir)
    project_root = os.path.dirname(visualization_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

PROJECT_ROOT = add_project_paths()

# --- Model Name Cleaning ---
# Centralized mappings for beautification
MODEL_NAME_MAPPINGS = {
    "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
    "phi3_14B_q4_medium-128k": "phi3 14B",
    # Add other specific mappings here if needed in the future
}

def clean_model_name(model_name: str) -> str:
    """
    Applies specific cleaning rules to a model name:
    1. Applies specific mappings first.
    2. If no specific mapping, removes '-128k' suffix.
    3. Replaces underscores '_' with spaces ' '.
    """
    # 1. Apply specific mapping
    cleaned_name = MODEL_NAME_MAPPINGS.get(model_name, model_name)

    # 2. If no specific mapping applied, apply general cleaning rules
    # Check if the specific mapping *changed* the name. If not, apply general rules.
    if cleaned_name == model_name: 
        # Remove '-128k' suffix
        if cleaned_name.endswith("-128k"):
            cleaned_name = cleaned_name.removesuffix("-128k")
        # Replace underscores with spaces
        cleaned_name = cleaned_name.replace("_", " ")

    return cleaned_name

# --- Other Utilities ---

def get_model_colors(model_names: List[str], config: Optional[Dict] = None) -> Optional[Dict[str, str]]:
    if config:
        model_color_map_from_config = config.get("visualization_settings", {}).get("MODEL_COLOR_MAP")
        if model_color_map_from_config:
            # Return colors only for models present in model_names
            return {m: c for m, c in model_color_map_from_config.items() if m in model_names}
    return None 

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters that are problematic in filenames."""
    # Replace common problematic characters with underscores
    sanitized = filename.replace(" ", "_").replace(":", "-").replace("/", "-").replace("\\", "-")
    # Remove any characters that are not alphanumeric, underscore, hyphen, or period
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ['_', '-', '.'])
    # Avoid starting with a period or hyphen
    if sanitized.startswith('.') or sanitized.startswith('-'):
        sanitized = '_' + sanitized
    return sanitized

def load_and_count_questions(filepath: str) -> int:
    """Loads a JSON file and returns the number of items in the root list."""
    if not os.path.exists(filepath):
        print(f"Error: Dataset file not found at {filepath}")
        return 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        else:
            print(f"Warning: Expected a list in {filepath}, but found {type(data)}. Cannot count items.")
            return 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        return 0