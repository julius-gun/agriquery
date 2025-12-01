import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from .plot_config import METRIC_DISPLAY_NAMES

def create_grouped_barchart(
    data: pd.DataFrame,
    output_path: str,
    metric_name: str,
    x_col: str,
    y_col: str,
    hue_col: str,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (10, 6),
    bar_label_fontsize: int = 8
) -> None:
    """
    Generates a generic grouped bar chart with direct labeling.
    """
    if data is None or data.empty:
        print(f"Warning: No data for {output_path}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None  # Remove error bars for cleaner visualization
    )

    # --- Direct Labeling ---
    # Loop through containers (groups of bars for each hue)
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.2f', 
            padding=3, 
            fontsize=bar_label_fontsize,
            color='black'
        )

    # Styling
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else METRIC_DISPLAY_NAMES.get(metric_name, metric_name), fontsize=12)
    
    # Rotate x-axis labels if many models are present
    plt.xticks(rotation=45, ha='right')
    
    # Y-axis limits (0 to 1.1 for scores)
    ax.set_ylim(0, 1.15)
    
    # Legend handling
    if hue_col:
        ax.legend(title=hue_col.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()
