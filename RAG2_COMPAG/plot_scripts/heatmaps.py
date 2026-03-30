import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
from plot_config import METRIC_DISPLAY_NAMES, FORMAT_ORDER

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
    Creates a single standardized heatmap.
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
        valid_index = [x for x in index_order if x in pivot_df.index]
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

def create_combined_heatmap(
    data_map: Dict[str, pd.DataFrame],
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
    figsize: Tuple[int, int] = (24, 8),
    annot_fmt: str = ".2f"
):
    """
    Creates a combined heatmap figure with 3 subplots (Markdown, JSON, XML).
    """
    if not data_map:
        return

    # Filter FORMAT_ORDER to only present data
    formats_to_plot = [fmt for fmt in FORMAT_ORDER if fmt in data_map]
    if not formats_to_plot:
        formats_to_plot = sorted(data_map.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create Subplots
    # We use width_ratios to give space for the colorbar if needed, 
    # but with sharey=True, standard subplots usually work well.
    fig, axes = plt.subplots(1, len(formats_to_plot), figsize=figsize, sharey=True)
    if len(formats_to_plot) == 1:
        axes = [axes]

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    for i, (ax, fmt) in enumerate(zip(axes, formats_to_plot)):
        df_fmt = data_map[fmt]
        
        # Pivot
        pivot_df = pd.pivot_table(
            df_fmt,
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc="mean"
        )
        
        # Apply Sorting (Crucial for alignment across subplots)
        if index_order:
            pivot_df = pivot_df.reindex(index_order)
        if columns_order:
            pivot_df = pivot_df.reindex(columns=columns_order)

        # Plot
        # Only show colorbar on the last plot
        is_last = (i == len(formats_to_plot) - 1)
        
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=annot_fmt,
            cmap=cmap,
            linewidths=0.5,
            linecolor="lightgray",
            cbar=is_last,
            cbar_kws={'label': metric_label} if is_last else None,
            vmin=0.0,
            vmax=1.0,
            ax=ax
        )
        
        ax.set_title(fmt, fontsize=16, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
            ax.tick_params(axis='y', rotation=0)
        else:
            ax.set_ylabel("")

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined heatmap: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()
