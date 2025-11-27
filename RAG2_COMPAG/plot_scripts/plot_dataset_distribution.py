import os
import sys
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# --- Adjust Python Path (using the helper) ---
# Assuming plot_utils.py exists in the same directory or is accessible
try:
    from plot_utils import add_project_paths

    PROJECT_ROOT = add_project_paths()  # Ensure project paths are set for imports below
except ImportError:
    print(
        "Warning: plot_utils.py not found or add_project_paths failed. Imports might fail."
    )
    # Attempt to add paths manually assuming standard structure
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../RAG2_COMPAG/plot_scripts
    project_root_dir = os.path.dirname(current_dir) # .../RAG2_COMPAG
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
        PROJECT_ROOT = project_root_dir
    print(f"Attempted to add project root: {project_root_dir}")


# --- Imports ---
try:
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"Error importing ConfigLoader: {e}")
    print(
        "Please ensure the script is run from a location where Python can find the 'utils' package"
    )
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def get_project_root() -> str:
    """Gets the absolute path to the project's root directory (RAG)."""
    # This is now redundant given PROJECT_ROOT from plot_utils, but keeping for logic consistency
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    return project_root


def load_and_count_questions(filepath: str) -> int:
    """Loads a JSON file and returns the number of items in the root list."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        else:
            print(
                f"Warning: Expected a list in {filepath}, but found {type(data)}. Cannot count items."
            )
            return 0
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return 0
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
    # Use PROJECT_ROOT determined at import time
    abs_config_path = os.path.join(
        PROJECT_ROOT, config_path
    )  # Ensure absolute path from root

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
        "unanswerable_questions": "Unanswerable",
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
        paths_to_check.append(abs_path)  # Keep track for reporting
        print(f"  - Processing '{label}' dataset from: {abs_path}")

        count = load_and_count_questions(abs_path)
        if count > 0:  # Only include datasets with questions found
            counts.append(count)
            labels.append(label)
        else:
            print(f"    -> No questions counted for '{label}'. Excluding from plot.")

    if not counts:
        print(
            "Error: No questions counted in any specified dataset. Cannot generate plot."
        )
        print(f"Checked paths: {paths_to_check}")
        return

    # 4. Generate Plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust size as needed

    # Use direct labeling (percentages on slices, labels for slices)
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",  # Format for percentages
        startangle=90,  # Start first slice at the top
        pctdistance=0.85,  # Position percentages inside slices
        textprops={"fontsize": 10},  # Adjust label font size if needed
        # wedgeprops={'edgecolor': 'white'} # Optional: add white border between slices
    )

    # Improve clarity of percentage labels
    plt.setp(
        autotexts, size=10, weight="bold", color="white"
    )  # Make percentages bold and white

    ax.set_title(
        "Distribution of Questions Across Datasets", fontsize=14, weight="bold"
    )
    # ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle. Already default for pie.

    # 5. Save Plot
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "question_dataset_distribution.png"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        plt.savefig(
            output_filepath, bbox_inches="tight"
        )  # Use bbox_inches='tight' to prevent label cutoff
        print(f"Successfully saved plot to: {output_filepath}")
    except Exception as e:
        print(f"Error saving plot to {output_filepath}: {e}")

    plt.close(fig)  # Close the figure to free memory
    print("--- Plot generation finished ---")


if __name__ == "__main__":
    # Assumes the script is run from the project root (e.g., 'p_llm_manual/RAG')
    # or that the path resolution works correctly via add_project_paths()

    # Define paths relative to the project root
    DEFAULT_CONFIG_REL_PATH = "config.json"
    DEFAULT_OUTPUT_REL_DIR = os.path.join("visualization", "plots")

    # Get project root (already calculated by import) and construct absolute paths for defaults
    # Re-using PROJECT_ROOT global from import
    
    default_config_abs_path = os.path.join(PROJECT_ROOT, DEFAULT_CONFIG_REL_PATH)
    default_output_abs_dir = os.path.join(PROJECT_ROOT, DEFAULT_OUTPUT_REL_DIR)

    # You could add argparse here if command-line overrides are needed
    config_to_use = default_config_abs_path
    output_dir_to_use = default_output_abs_dir

    plot_question_distribution(config_path=config_to_use, output_dir=output_dir_to_use)
