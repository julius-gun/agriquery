# visualization/plot_scripts/plot_utils.py
import re
import os
import sys

def sanitize_filename(name: str) -> str:
    """Removes or replaces characters problematic for filenames."""
    # Remove or replace characters like ':', '/', '\', etc.
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    # Replace sequences of whitespace or underscores with a single underscore
    name = re.sub(r'[\s_]+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name if name else "invalid_name"

# --- Helper to Adjust Python Path ---
# This helps scripts in plot_scripts find modules in visualization and the project root (RAG)
def add_project_paths():
    """Adds project root and visualization directory to sys.path if not already present."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.dirname(current_dir) # Parent of plot_scripts
    project_root_dir = os.path.dirname(visualization_dir) # Parent of visualization (RAG)

    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    if visualization_dir not in sys.path:
        sys.path.insert(1, visualization_dir)

# Call this function immediately so paths are set when modules are imported
add_project_paths()

# --- Constants (Example - could be moved or expanded) ---
# Define constants used by multiple plotting scripts if needed
# e.g., DEFAULT_OUTPUT_DIR = os.path.join(visualization_dir, "plots")
# e.g., DEFAULT_RESULTS_DIR = os.path.join(project_root_dir, "results")