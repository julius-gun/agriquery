import matplotlib.pyplot as plt
import os

# --- Constants & Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Target: LLM_Research/visualization/plots
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../LLM_Research/visualization/plots"))

def plot_transfer_times():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Measured data from ETP transfer tests (English Manual + Compressed Image)
    # Values provided in Appendix/Prompt:
    # English MD: 47.0s, English XML: 49.3s, English JSON: 82.7s, Image: 5.8s
    data = [
        {"label": "Markdown", "time": 47.0, "color": "#4c72b0"},
        {"label": "XML",      "time": 49.3, "color": "#4c72b0"},
        {"label": "JSON",     "time": 82.7, "color": "#4c72b0"},
        {"label": "1x Image\n(39.9 KB)", "time": 5.8,  "color": "#dd8452"}
    ]
    
    labels = [item["label"] for item in data]
    values = [item["time"] for item in data]
    colors = [item["color"] for item in data]

    # Console Output for Verification
    print(f"{'Item':<20} | {'Time (s)':<10}")
    print("-" * 35)
    for lbl, val in zip(labels, values):
        clean_lbl = lbl.replace('\n', ' ')
        print(f"{clean_lbl:<20} | {val:<10.1f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, values, color=colors, width=0.6)
    
    # Direct Labeling (No Legend)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Labels and Formatting
    ax.set_ylabel('Measured Transfer Time (seconds)', fontsize=12)
    ax.set_title('Data Transfer Time over ISOBUS (ETP)', fontsize=14)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # Ensure grid is behind bars
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "bar_transfer_time_format.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    plot_transfer_times()
