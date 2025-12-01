import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from plot_config import METRIC_DISPLAY_NAMES, FORMAT_ORDER

def create_combined_scatter(
    data_map: Dict[str, pd.DataFrame],
    output_path: str,
    metric_name: str,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    hue_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (20, 7)
) -> None:
    """
    Generates a combined figure with 3 subplots (Markdown, JSON, XML) for comparison.
    """
    if not data_map:
        print(f"Warning: No data map for {output_path}")
        return

    # Determine Global Limits for consistent scales across subplots
    all_x = []
    all_y = []
    for df in data_map.values():
        all_x.extend(df[x_col].tolist())
        all_y.extend(df[y_col].tolist())
    
    if not all_x: return

    min_val = min(min(all_x), min(all_y))
    max_val = max(max(all_x), max(all_y))
    
    # Add padding
    pad = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    limit_min = min_val - pad
    limit_max = max_val + pad

    # Filter FORMAT_ORDER to only present data
    formats_to_plot = [fmt for fmt in FORMAT_ORDER if fmt in data_map]
    if not formats_to_plot:
        formats_to_plot = sorted(data_map.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create Subplots
    fig, axes = plt.subplots(1, len(formats_to_plot), figsize=figsize, sharey=True, sharex=True)
    
    # Handle single subplot case just in case
    if len(formats_to_plot) == 1:
        axes = [axes]

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    for ax, fmt in zip(axes, formats_to_plot):
        df_plot = data_map[fmt]
        
        # Scatter
        sns.scatterplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue=label_col,
            style=label_col,
            hue_order=hue_order, # Consistent colors across subplots
            style_order=hue_order,
            s=150,
            palette="tab10",
            legend=False,
            ax=ax
        )
        
        # Diagonal Line
        ax.plot(
            [limit_min, limit_max], 
            [limit_min, limit_max], 
            ls="--", c=".6", label="Perfect Balance (y=x)"
        )
        
        # Labels
        for i, row in df_plot.iterrows():
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
        ax.set_title(fmt, fontsize=16, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        # Only set Y label for the first plot
        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel("")

        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(title, fontsize=20, y=1.05)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined scatter: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()
