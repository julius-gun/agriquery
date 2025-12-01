import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from plot_config import METRIC_DISPLAY_NAMES

def create_heatmap(
    data: pd.DataFrame,
    output_path: str,
    index_col: str,
    columns_col: str,
    values_col: str,
    metric_name: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    index_order: Optional[List[str]] = None,
    columns_order: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 8),
    annot_fmt: str = ".2f"
):
    """
    Creates a standardized heatmap.
    """
    if data is None or data.empty:
        print(f"Warning: No data for heatmap {output_path}")
        return

    # Pivot Data
    try:
        pivot_df = pd.pivot_table(
            data,
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc="mean"
        )
    except Exception as e:
        print(f"Error pivoting data for {output_path}: {e}")
        return

    # Apply Sorting
    if index_order:
        # Filter order to items present in data
        valid_index = [x for x in index_order if x in pivot_df.index]
        # Append remaining items
        remaining = [x for x in pivot_df.index if x not in valid_index]
        pivot_df = pivot_df.reindex(valid_index + sorted(remaining))
    
    if columns_order:
        valid_cols = [x for x in columns_order if x in pivot_df.columns]
        remaining = [x for x in pivot_df.columns if x not in valid_cols]
        pivot_df = pivot_df[valid_cols + sorted(remaining)]

    # Plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=figsize)

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    sns.heatmap(
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

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel if xlabel else columns_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(ylabel if ylabel else index_col.replace("_", " ").title(), fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()
