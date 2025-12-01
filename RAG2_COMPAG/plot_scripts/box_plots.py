import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from .plot_config import METRIC_DISPLAY_NAMES

def create_boxplot(
    data: pd.DataFrame,
    output_path: str,
    x_col: str,
    y_col: str,
    metric_name: str,
    title: str,
    hue_col: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Generates a standardized box plot with overlaid strip plot.
    """
    if data is None or data.empty:
        print(f"Warning: No data for boxplot {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=figsize)

    # Boxplot
    sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col if hue_col else x_col, # If no hue, color by X
        order=order,
        hue_order=hue_order,
        palette=palette,
        dodge=False if not hue_col else True,
        legend=False # Custom legend handling
    )

    # Strip plot for individual points
    sns.stripplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col if hue_col else None,
        order=order,
        hue_order=hue_order,
        size=4,
        color=".3" if not hue_col else None,
        jitter=True,
        dodge=False if not hue_col else True,
        legend=False
    )

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.ylim(0, 1.1)
    
    plt.xticks(rotation=45, ha='right')

    # Add Legend if Hue is used (and strictly for Language to keep it clean)
    if hue_col:
        # Create custom handles for legend to avoid duplicate entries from strip/box
        import matplotlib.patches as mpatches
        if palette and hue_order:
            handles = [mpatches.Patch(color=palette[h], label=h.title()) for h in hue_order if h in palette]
            plt.legend(handles=handles, title=hue_col.title(), bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved boxplot: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()
