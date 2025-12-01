import os
import sys
import re

def add_project_paths() -> str:
    """
    Adds the project root directory to sys.path to allow imports from 
    sibling directories (e.g., utils).
    Returns the absolute path to the project root.
    """
    # Current: .../RAG2_COMPAG/plot_scripts/plot_utils.py
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Root: .../RAG2_COMPAG
    project_root = os.path.dirname(current_script_dir)
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    return project_root

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a string for use as a filename, removing problematic characters.
    """
    # Replace separators with hyphens
    sanitized = filename.replace(" ", "_").replace(":", "-").replace("/", "-").replace("\\", "-")
    # Keep only alphanumeric, underscores, hyphens, and dots
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '', sanitized)
    # Prevent hidden files
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized
        
    return sanitized
