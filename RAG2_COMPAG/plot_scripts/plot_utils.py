# visualization/plot_scripts/plot_utils.py
import os
import sys
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# --- Path Setup ---
# Calculate necessary paths relative to this script's location
# Ensure this runs *before* imports that depend on the project root path.
def add_project_paths() -> str:
    """Adds the project root directory (RAG) to sys.path and returns it."""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # New structure: RAG2_COMPAG/plot_scripts/plot_utils.py
    # Project root is the parent of plot_scripts
    project_root = os.path.dirname(current_script_dir)
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # print(f"DEBUG: Added Project Root to sys.path: {project_root}") # Optional debug
    return project_root

PROJECT_ROOT = add_project_paths() # Execute path setup and store the root path

# --- Imports ---
# Now that PROJECT_ROOT is (hopefully) in sys.path, these should work
try:
    from utils.config_loader import ConfigLoader # This import is present in the user-provided plot_utils.py, though not directly used by functions here.
except ImportError as e:
    print(f"Error importing required modules after path setup: {e}")
    print(f"Project Root determined: {PROJECT_ROOT}")
    print(f"Current sys.path: {sys.path}")
    print("Please ensure 'utils/config_loader.py' exists relative to the project root.")
    sys.exit(1)
# --- End Imports ---

def get_model_colors(model_names: List[str], config: Optional[Dict] = None) -> Optional[Dict[str, str]]:
    if config:
        model_color_map_from_config = config.get("visualization_settings", {}).get("MODEL_COLOR_MAP")
        if model_color_map_from_config:
            # Return colors only for models present in model_names
            return {m: c for m, c in model_color_map_from_config.items() if m in model_names}
    # Fallback or more sophisticated color generation can be added here if no map in config
    return None # Or generate a default palette using seaborn for the given model_names

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters that are problematic in filenames."""
    # Replace common problematic characters with underscores
    sanitized = filename.replace(" ", "_").replace(":", "-").replace("/", "-").replace("\\", "-")
    # Remove any characters that are not alphanumeric, underscore, hyphen, or period
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ['_', '-', '.'])
    # Avoid starting with a period or hyphen
    if sanitized.startswith('.') or sanitized.startswith('-'):
        sanitized = '_' + sanitized
    # Limit length if necessary (optional)
    # max_len = 200
    # sanitized = sanitized[:max_len]
    return sanitized

def load_and_count_questions(filepath: str) -> int:
    """Loads a JSON file and returns the number of items in the root list."""
    # Check if path exists before opening
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
    # Removed FileNotFoundError as it's checked above
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        return 0

def plot_question_distribution(config_path: str, output_dir: str):
    """
    Loads question dataset paths from config, counts questions in each,
    and generates a pie chart showing the distribution.
    """
    print("--- Generating Question Distribution Pie Chart ---")
    # Use PROJECT_ROOT calculated at the start
    abs_config_path = os.path.join(PROJECT_ROOT, config_path) # Ensure absolute path from root

    # 1. Load Config
    try:
        config_loader = ConfigLoader(abs_config_path)
        dataset_paths_config = config_loader.config.get("question_dataset_paths")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{abs_config_path}'")
        return
    except Exception as e:
        print(f"Error loading config file '{abs_config_path}': {e}")
        return

    if not dataset_paths_config:
        print("Error: 'question_dataset_paths' not found in the configuration.")
        return

    # 2. Define datasets and friendly labels
    datasets_to_plot = {
        "general_questions": "General",
        "table_questions": "Table",
        "unanswerable_questions": "Unanswerable"
    }

    counts = []
    labels = []
    paths_to_check = []

    # 3. Load data and count questions
    print("Counting questions in datasets:")
    for key, label in datasets_to_plot.items():
        rel_path = dataset_paths_config.get(key)
        if not rel_path:
            print(f"Warning: Path for '{key}' not found in config. Skipping.")
            continue

        # Construct absolute path from project root
        abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
        paths_to_check.append(abs_path) # Keep track for reporting
        print(f"  - Processing '{label}' dataset from: {abs_path}")

        count = load_and_count_questions(abs_path)
        if count > 0: # Only include datasets with questions found
            counts.append(count)
            labels.append(label)
        else:
            print(f"    -> No questions counted for '{label}'. Excluding from plot.")

    if not counts:
        print("Error: No questions counted in any specified dataset. Cannot generate plot.")
        print(f"Checked paths: {paths_to_check}")
        return

    # 4. Generate Plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(8, 8)) # Adjust size as needed

    # Use direct labeling (percentages on slices, labels for slices)
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%', # Format for percentages
        startangle=90,     # Start first slice at the top
        pctdistance=0.85,  # Position percentages inside slices
        textprops={'fontsize': 10}, # Adjust label font size if needed
        # wedgeprops={'edgecolor': 'white'} # Optional: add white border between slices
    )

    # Improve clarity of percentage labels
    plt.setp(autotexts, size=10, weight="bold", color="white") # Make percentages bold and white

    # ax.set_title('Distribution of Questions Across Datasets', fontsize=14, weight='bold')

    # 5. Save Plot
    # Ensure output dir is absolute from project root if needed, or relative as before
    abs_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    output_filename = "question_dataset_distribution.png"
    output_filepath = os.path.join(abs_output_dir, output_filename)

    try:
        plt.savefig(output_filepath, bbox_inches='tight') # Use bbox_inches='tight' to prevent label cutoff
        print(f"Successfully saved plot to: {output_filepath}")
    except Exception as e:
        print(f"Error saving plot to {output_filepath}: {e}")

    plt.close(fig) # Close the figure to free memory
    print("--- Plot generation finished ---")


if __name__ == "__main__":
    # Define paths relative to the project root (PROJECT_ROOT)
    # PROJECT_ROOT should be 'p_llm_manual/RAG'
    DEFAULT_CONFIG_REL_PATH = "config.json" # Relative to PROJECT_ROOT
    DEFAULT_OUTPUT_REL_DIR = os.path.join("visualization", "plots") # Relative to PROJECT_ROOT

    # Use the relative paths directly; the functions construct absolute paths using PROJECT_ROOT
    config_to_use = DEFAULT_CONFIG_REL_PATH
    output_dir_to_use = DEFAULT_OUTPUT_REL_DIR

    plot_question_distribution(config_path=config_to_use, output_dir=output_dir_to_use)