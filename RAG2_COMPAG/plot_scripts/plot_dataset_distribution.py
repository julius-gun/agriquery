import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Path setup
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
    from plot_scripts.plot_config import METRIC_DISPLAY_NAMES # Just to ensure config is loaded/style set
except ImportError:
    ConfigLoader = None

def load_count(filepath: str) -> int:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0

def plot_question_distribution(config_path: str, output_dir: str):
    """Generates a pie chart for question distribution."""
    print("\n--- Generating Question Distribution Plot ---")
    
    abs_config = os.path.join(project_root_dir, config_path)
    if not os.path.exists(abs_config) or not ConfigLoader:
        print("Config not found or loader missing.")
        return

    loader = ConfigLoader(abs_config)
    paths = loader.config.get("question_dataset_paths", {})
    
    labels = {
        "general_questions": "General",
        "table_questions": "Table",
        "unanswerable_questions": "Unanswerable"
    }

    data_counts = []
    data_labels = []

    for key, label in labels.items():
        rel = paths.get(key)
        if rel:
            count = load_count(os.path.join(project_root_dir, rel))
            if count > 0:
                data_counts.append(count)
                data_labels.append(label)

    if not data_counts:
        print("No data found.")
        return

    # Plot
    sns.set_theme(style="white") # Pie charts look better without grid
    plt.figure(figsize=(8, 8))
    
    plt.pie(
        data_counts,
        labels=data_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel"),
        textprops={'fontsize': 12}
    )
    
    plt.title("Question Distribution by Dataset", fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_question_distribution("config.json", os.path.join(project_root_dir, "visualization", "plots"))
