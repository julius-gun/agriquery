import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from plot_config import METRIC_DISPLAY_NAMES

def create_comparison_scatter(
    data: pd.DataFrame,
    output_path: str,
    x_col: str,
    y_col: str,
    label_col: str,
    metric_name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[float, float] = (8, 8) # Square figure size
) -> None:
    """
    Generates a scatter plot comparing two dimensions (e.g., English vs Non-English).
    Includes a diagonal reference line (y=x) to show balance.
    """
    if data is None or data.empty:
        print(f"Warning: No data for scatter {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Force square figure to help with aspect ratio
    plt.figure(figsize=figsize)

    # Determine limits to draw the line across the whole plot
    # We want a square domain (min_val to max_val on both axes)
    all_vals = pd.concat([data[x_col], data[y_col]])
    min_val = all_vals.min()
    max_val = all_vals.max()
    
    # Add padding (10% of range)
    pad = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    limit_min = min_val - pad
    limit_max = max_val + pad

    # Scatter Plot
    ax = sns.scatterplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=label_col,
        style=label_col,
        s=150, # Marker size
        palette="tab10",
        legend=False 
    )

    # --- Add Diagonal Line (Perfect Balance) ---
    plt.plot(
        [limit_min, limit_max], 
        [limit_min, limit_max], 
        ls="--", c=".6", label="Perfect Balance (y=x)"
    )

    # --- Direct Labeling ---
    # Annotate each point with the model name
    for i, row in data.iterrows():
        ax.annotate(
            row[label_col],
            (row[x_col], row[y_col]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=9,
            alpha=0.9,
            fontweight='medium'
        )

    # Styling
    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    plt.title(title, fontsize=15, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Enforce Square Aspect Ratio and Limits
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.set_aspect('equal', adjustable='box')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved scatter plot: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()
