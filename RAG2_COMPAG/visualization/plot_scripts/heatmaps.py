# visualization/plot_scripts/heatmaps.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, Tuple, List, Dict, Any

def create_heatmap(
    data: pd.DataFrame,
    output_path: str,
    index_col: str,
    columns_col: str,
    values_col: str,
    metric_label: str,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    annot_fmt: str = ".3f",
    figsize: Tuple[int, int] = (14, 8),
    sort_index: bool = True,
    sort_columns: bool = True,
):
    """
    Creates a generic heatmap from a DataFrame.
    
    Args:
        data: DataFrame containing the data.
        output_path: Path to save the image.
        index_col: Column for Y-axis (e.g., 'question_model').
        columns_col: Column for X-axis (e.g., 'language').
        values_col: Column for cell values (e.g., 'metric_value').
        metric_label: Label for the colorbar/metric (e.g., 'F1 Score').
        title: Chart title.
    """
    if data is None or data.empty:
        print(f"Warning: No data for heatmap '{output_path}'.")
        return

    # Pivot the data
    try:
        pivot_df = pd.pivot_table(
            data,
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc="mean" # Average if multiple entries exist for a cell
        )
    except Exception as e:
        print(f"Error creating pivot table for {output_path}: {e}")
        return

    if pivot_df.empty:
        print(f"Warning: Pivot table is empty for {output_path}.")
        return

    # Sorting
    if sort_index:
        pivot_df = pivot_df.sort_index()
    if sort_columns:
        # For columns, often nice to sort by average value if no natural order
        # Here we default to alphabetical/natural sort, can be enhanced
        pivot_df = pivot_df.sort_index(axis=1)

    # Plotting
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="white")
    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={'label': metric_label},
        vmin=0.0,
        vmax=1.0,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel(xlabel if xlabel else columns_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(ylabel if ylabel else index_col.replace("_", " ").title(), fontsize=12)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap: {output_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    plt.close()
