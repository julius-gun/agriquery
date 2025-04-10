# visualization/plot_scripts/box_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List # Added List

def create_f1_boxplot(
    data: pd.DataFrame,
    group_by_column: str,
    output_path: str,
    score_column: str = 'f1_score',
    hue_column: Optional[str] = None, # Added: Column to use for coloring points
    hue_order: Optional[List[str]] = None, # Added: Order for hue levels (legend)
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figure_width: int = 12,
    figure_height: int = 7
):
    """
    Creates a box plot of F1 scores grouped by a specified column.
    Optionally colors individual points based on a hue column (e.g., language).

    Args:
        data: Pandas DataFrame containing the data.
        group_by_column: The name of the column to group the data by (x-axis).
        output_path: The full path where the plot image will be saved.
        score_column: The name of the column containing the scores to plot (y-axis).
        hue_column: Optional name of the column to use for coloring individual points.
        hue_order: Optional list specifying the order of categories in the hue legend.
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
    # Check hue column if provided
    use_hue = False
    if hue_column:
        if hue_column not in data.columns:
            print(f"Warning: Hue column '{hue_column}' not found in DataFrame. Plotting without hue.")
        else:
            use_hue = True
            print(f"Using column '{hue_column}' for point colors.")


    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir: # Only create if output_path includes a directory part
        os.makedirs(output_dir, exist_ok=True)

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(figure_width, figure_height))

    # Generate the boxplot - Simplified: Removed hue and legend from boxplot itself
    ax = sns.boxplot(
        x=group_by_column,
        y=score_column,
        hue=group_by_column, # Assign x variable to hue
        data=data,
        palette="viridis",
        # hue=group_by_column, # Removed: Not needed for the boxes themselves
        # legend=False # Removed
    )

    # --- Add individual data points ---
    stripplot_kwargs = {
        "x": group_by_column,
        "y": score_column,
        "data": data,
        "size": 4,
        "jitter": True,
        "ax": ax,
    }

    if use_hue:
        stripplot_kwargs.update({
            "hue": hue_column,
            "hue_order": hue_order, # Pass hue_order if provided
            "dodge": True, # Separate points by hue within each group
            "legend": "auto" # Let seaborn handle the legend
            # Removed 'color=".25"' as hue will control color
        })
        # Adjust legend title if needed (optional)
        # try:
        #     handles, labels = ax.get_legend_handles_labels()
        #     # Modify labels or title here if necessary
        #     ax.legend(handles=handles, labels=labels, title=hue_column.replace("_", " ").title())
        # except AttributeError: # Handle cases where legend might not be generated
        #     pass

    else:
        # Original stripplot settings when not using hue
        stripplot_kwargs.update({
            "color": ".25",
            "legend": False # No legend needed if no hue
        })

    sns.stripplot(**stripplot_kwargs)
    # --- End stripplot ---


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

    # Adjust legend position if it exists and overlaps (optional refinement)
    if use_hue and ax.get_legend() is not None:
         # Example: Move legend outside plot area
         # ax.legend(title=hue_column.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc='upper left')
         # Or let tight_layout try first
         pass # Keep default legend placement for now


    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight') # Added bbox_inches='tight' to help fit legend
        print(f"Box plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

    plt.close() # Close the plot figure to free memory
