# visualization/plot_scripts/box_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional

def create_f1_boxplot(
    data: pd.DataFrame,
    group_by_column: str,
    output_path: str,
    score_column: str = 'f1_score',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figure_width: int = 12,
    figure_height: int = 7
):
    """
    Creates a box plot of F1 scores grouped by a specified column.

    Args:
        data: Pandas DataFrame containing the data (must include group_by_column and score_column).
        group_by_column: The name of the column to group the data by (e.g., 'question_model', 'language').
        output_path: The full path where the plot image will be saved.
        score_column: The name of the column containing the scores to plot (default: 'f1_score').
        title: Optional title for the plot.
        xlabel: Optional label for the x-axis.
        ylabel: Optional label for the y-axis.
        figure_width: Width of the plot figure in inches.
        figure_height: Height of the plot figure in inches.
    """
    if data is None or data.empty:
        print("Error: Cannot create plot. Input data is None or empty.")
        return
    if group_by_column not in data.columns:
        print(f"Error: Grouping column '{group_by_column}' not found in DataFrame.")
        return
    if score_column not in data.columns:
        print(f"Error: Score column '{score_column}' not found in DataFrame.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir: # Only create if output_path includes a directory part
        os.makedirs(output_dir, exist_ok=True)

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(figure_width, figure_height))

    # Generate the boxplot - **MODIFIED LINE**
    # Assign group_by_column to hue and disable legend to avoid FutureWarning
    ax = sns.boxplot(
        x=group_by_column,
        y=score_column,
        hue=group_by_column, # Assign x variable to hue
        data=data,
        palette="viridis",
        legend=False # Disable the legend as it's redundant with x-axis labels
    )

    # Add individual data points for better visibility when few points per box
    # Keep stripplot as is, or optionally assign hue here too if desired, but usually not needed
    sns.stripplot(
        x=group_by_column,
        y=score_column,
        data=data,
        color=".25",
        size=4,
        jitter=True,
        ax=ax,
        legend=False # Ensure legend is off here too if hue were added
    )


    # Improve layout and labels
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability

    # Set title and labels
    plot_title = title or f'F1 Score Distribution by {group_by_column.replace("_", " ").title()}'
    x_axis_label = xlabel or group_by_column.replace("_", " ").title()
    y_axis_label = ylabel or score_column.replace("_", " ").title()

    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.ylim(0, 1.05) # Set y-axis limits for F1 score (0 to 1) with a little padding

    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300) # Save with higher resolution
        print(f"Box plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

    plt.close() # Close the plot figure to free memory
