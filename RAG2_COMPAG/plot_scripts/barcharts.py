import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from plot_config import METRIC_DISPLAY_NAMES, LANGUAGE_CODES, FORMAT_CODES

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
    bar_label_fontsize: int = 8,
    ylim: Optional[Tuple[float, float]] = (0, 1.15),
    value_format: str = '%.2f'
) -> None:
    """
    Generates a generic grouped bar chart with direct labeling inside bars.
    
    Args:
        ylim: Y-axis limits. Defaults to (0, 1.15) for 0-1 metrics. 
              Pass None for auto-scaling (e.g., token counts).
        value_format: Format string for bar labels. Defaults to '%.2f'.
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
        errorbar=None
    )

    # --- Direct Labeling ---
    
    # 1. Inner Labels (Series Name)
    if hue_order:
        for i, container in enumerate(ax.containers):
            if i < len(hue_order):
                series_label = hue_order[i]
                short_label = LANGUAGE_CODES.get(series_label)
                if not short_label:
                    short_label = FORMAT_CODES.get(series_label, series_label)

                ax.bar_label(
                    container,
                    labels=[short_label] * len(container),
                    label_type='center',
                    rotation=90,
                    fontsize=bar_label_fontsize,
                    color='white',
                    fontweight='bold'
                )

    # 2. Outer Labels (Values)
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=value_format, 
            padding=3, 
            fontsize=bar_label_fontsize,
            color='black'
        )

    # Styling
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else METRIC_DISPLAY_NAMES.get(metric_name, metric_name), fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()


def create_bidirectional_barchart(
    data: pd.DataFrame,
    output_path: str,
    x_col: str,
    y_col_top: str,
    y_col_bottom: str,
    hue_col: str,
    top_label: str = "Top Metric",
    bottom_label: str = "Bottom Metric",
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    title: str = "",
    xlabel: str = "",
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Generates a mirrored (bidirectional) bar chart.
    Useful for comparing Tokens (Top) vs Characters (Bottom).
    """
    if data is None or data.empty:
        print(f"Warning: No data for {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create copies to avoid modifying original dataframe outside
    df = data.copy()
    
    # Ensure bottom metric is negative for plotting
    # We assume y_col_bottom in the input DF is positive; we flip it here.
    df['__neg_bottom'] = -df[y_col_bottom].abs()

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # --- Plot Top (Positive) ---
    sns.barplot(
        data=df, x=x_col, y=y_col_top, hue=hue_col,
        order=x_order, hue_order=hue_order, palette=palette,
        ax=ax, errorbar=None
    )

    # --- Plot Bottom (Negative) ---
    sns.barplot(
        data=df, x=x_col, y='__neg_bottom', hue=hue_col,
        order=x_order, hue_order=hue_order, palette=palette,
        ax=ax, errorbar=None
    )

    # --- Formatting Y-Axis to show positive numbers ---
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f'{abs(int(x)):,}'

    ax.yaxis.set_major_formatter(major_formatter)

    # --- Add Horizontal Zero Line ---
    ax.axhline(0, color='black', linewidth=0.8)

    # --- Annotations for Top/Bottom Areas ---
    # Get y-limits to place text appropriately
    y_min, y_max = ax.get_ylim()
    
    # Label "Tokens" (Top Left)
    ax.text(
        x=-0.4, y=y_max * 0.95, s=top_label, 
        fontsize=12, fontweight='bold', ha='left', va='top', 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    # Label "Characters" (Bottom Left)
    ax.text(
        x=-0.4, y=y_min * 0.95, s=bottom_label, 
        fontsize=12, fontweight='bold', ha='left', va='bottom', 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    # --- Bar Labels (Inner Code & Outer Value) ---
    
    # We iterate containers. Usually:
    # First half of containers = Top bars (per hue)
    # Second half of containers = Bottom bars (per hue)
    
    containers = ax.containers
    mid_point = len(containers) // 2
    
    for i, container in enumerate(containers):
        is_top = i < mid_point
        
        # 1. Inner Label (Format Name) - Only on Top Bars to reduce clutter
        if is_top and hue_order:
            hue_idx = i
            if hue_idx < len(hue_order):
                series_label = hue_order[hue_idx]
                short_label = FORMAT_CODES.get(series_label, series_label)
                
                ax.bar_label(
                    container,
                    labels=[short_label] * len(container),
                    label_type='center',
                    rotation=90,
                    fontsize=9,
                    color='white',
                    fontweight='bold'
                )

        # 2. Outer Label (Value)
        # For bottom bars, the values in 'container' are negative.
        # We want to label them as absolute integers.
        labels = [f'{abs(int(v)):,}' if v != 0 else '' for v in container.datavalues]
        
        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=7,
            color='black'
        )

    # Styling
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    # Remove legend (handled by labels)
    if ax.get_legend():
        ax.get_legend().remove()
        
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved bidirectional plot: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    finally:
        plt.close()