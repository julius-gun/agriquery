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
    sort_by_median_score: bool = True, # New parameter: Control sorting
    hue_column: Optional[str] = None,
    hue_order: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figure_width: int = 12,
    figure_height: int = 7
):
    """
    Creates a box plot of scores grouped by a specified column.
    By default, groups are ordered by their median score in ascending order.
    Optionally colors individual points based on a hue column.

    Args:
        data: Pandas DataFrame containing the data.
        group_by_column: The name of the column to group the data by (x-axis).
                         The categories in this column will be ordered by median score.
        output_path: The full path where the plot image will be saved.
        score_column: The name of the column containing the scores to plot (y-axis).
        sort_by_median_score: If True (default), order the groups on the x-axis
                              by their median score (ascending). If False, use
                              default ordering.
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

    # --- Determine order based on median score (optional) ---
    ordered_groups = None # Default to no specific order
    if sort_by_median_score:
        try:
            # Ensure score_column is numeric before calculating median
            if pd.api.types.is_numeric_dtype(data[score_column]):
                median_scores = data.groupby(group_by_column)[score_column].median()
                # Sort the groups based on median score in ascending order
                ordered_groups = median_scores.sort_values(ascending=True).index.tolist()
                print(f"Ordering '{group_by_column}' by median '{score_column}' (ascending): {ordered_groups}")
            else:
                print(f"Warning: Score column '{score_column}' is not numeric. Cannot sort by median. Plotting in default order.")
        except KeyError:
            print(f"Warning: Could not group by '{group_by_column}' or find score column '{score_column}' for ordering. Plotting in default order.")
        except Exception as e:
             print(f"Warning: An error occurred during group ordering: {e}. Plotting in default order.")
    else:
        print(f"Plotting '{group_by_column}' in default order (sorting by score disabled).")
    # --- End order calculation ---


    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(figure_width, figure_height))

    # Generate the boxplot - Pass 'order' if calculated
    ax = sns.boxplot(
        x=group_by_column,
        y=score_column,
        hue=group_by_column, # Assign x variable to hue for box colors
        data=data,
        palette="viridis",
        order=ordered_groups, # Apply the calculated order (or None for default)
        legend=False
    )

    # --- Add individual data points ---
    stripplot_kwargs = {
        "x": group_by_column,
        "y": score_column,
        "data": data,
        "size": 5, # Slightly larger points might be good
        "jitter": True,
        "ax": ax,
        "order": ordered_groups, # Apply the same order (or None) to stripplot
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
    title_suffix = " (Ordered Ascending by Median Score)" if sort_by_median_score and ordered_groups else ""
    plot_title = title or f'{score_column.replace("_", " ").title()} Distribution by {group_by_column.replace("_", " ").title()}{title_suffix}'
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
         # Update legend title if hue is used
         try:
             current_legend = ax.get_legend()
             if current_legend:
                 # Make legend title more readable
                 legend_title = hue_column.replace("_", " ").title()
                 current_legend.set_title(legend_title)
                 # Optionally move legend outside
                 # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=legend_title)
         except AttributeError:
             pass # No legend to modify

    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot
    try:
        # Use bbox_inches='tight' to try and fit legend if moved outside
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

    plt.close() # Close the plot figure to free memory


# --- New Function ---
def create_dataset_success_boxplot(
    data: pd.DataFrame,
    output_path: str,
    retrieval_algorithm: str,
    language: str, # Added parameter
    chunk_size: int, # Added parameter
    overlap_size: int, # Added parameter
    sort_by_median_score: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figure_width: int = 12,
    figure_height: int = 7
):
    """
    Creates a box plot of dataset success rates grouped by question model,
    with individual points colored by dataset type. Includes specific run
    parameters in the title.

    Assumes input data is pre-filtered for a specific retrieval algorithm,
    language, chunk size, overlap size, and metric_type='dataset_success'.

    Args:
        data: Pandas DataFrame containing the filtered data. Must include
              'question_model', 'metric_value', and 'dataset_type' columns.
        output_path: The full path where the plot image will be saved.
        retrieval_algorithm: The retrieval algorithm used.
        language: The language used for this data.
        chunk_size: The chunk size used.
        overlap_size: The overlap size used.
        sort_by_median_score: If True (default), order models on the x-axis
                              by their median success rate (ascending).
        title: Optional title for the plot. If None, a default is generated.
        xlabel: Optional label for the x-axis. Defaults to 'Question Model'.
        ylabel: Optional label for the y-axis. Defaults to 'Dataset Success Rate'.
        figure_width: Width of the plot figure in inches.
        figure_height: Height of the plot figure in inches.
    """
    group_by_column = 'question_model'
    score_column = 'metric_value' # From the new data extractor structure
    hue_column = 'dataset_type' # Color points by dataset type

    if data is None or data.empty:
        print(f"Error: Cannot create plot for {retrieval_algorithm}. Input data is None or empty.")
        return
    required_columns = [group_by_column, score_column, hue_column]
    if not all(col in data.columns for col in required_columns):
        print(f"Error: Data for {retrieval_algorithm} is missing one or more required columns: {required_columns}.")
        return
    if not pd.api.types.is_numeric_dtype(data[score_column]):
        print(f"Error: Score column '{score_column}' is not numeric for {retrieval_algorithm}. Cannot plot.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Determine order of models based on median score (optional) ---
    ordered_groups = None
    if sort_by_median_score:
        try:
            # Group by model, calculate median success rate across all its dataset points
            median_scores = data.groupby(group_by_column)[score_column].median()
            ordered_groups = median_scores.sort_values(ascending=True).index.tolist()
            print(f"Ordering '{group_by_column}' by median '{score_column}' (ascending) for {retrieval_algorithm}: {ordered_groups}")
        except Exception as e:
             print(f"Warning: An error occurred during group ordering for {retrieval_algorithm}: {e}. Plotting in default order.")
    else:
        print(f"Plotting '{group_by_column}' in default order for {retrieval_algorithm} (sorting by score disabled).")
    # --- End order calculation ---

    # --- Determine hue order for datasets (optional but good for consistency) ---
    # Get unique dataset types present in the data and sort them alphabetically
    hue_order = sorted(data[hue_column].unique())
    print(f"Using dataset types for hue (legend order): {hue_order}")
    # --- End hue order ---

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(figure_width, figure_height))

    # 1. Generate the boxplot (boxes colored by model)
    ax = sns.boxplot(
        x=group_by_column,
        y=score_column,
        hue=group_by_column, # Color boxes by model
        data=data,
        palette="viridis", # Or another palette
        order=ordered_groups,
        legend=False # No legend for the boxes
    )

    # 2. Add individual data points (colored by dataset_type)
    sns.stripplot(
        x=group_by_column,
        y=score_column,
        hue=hue_column, # Color points by dataset type
        hue_order=hue_order, # Use consistent dataset order
        data=data,
        size=5,
        jitter=True,
        dodge=True, # Separate points by dataset type
        order=ordered_groups,
        ax=ax,
        legend="auto" # Create legend for dataset types
    )

    # Improve layout and labels
    plt.xticks(rotation=45, ha='right')

    # Set title and labels
    # title_suffix = " (Ordered by Median Success Rate)" if sort_by_median_score and ordered_groups else ""
    title_suffix = ""
    # Include algo, lang, chunk, overlap in the default title
    default_title = (
        f'Dataset Success Rate by Model\n'
        f'({retrieval_algorithm.capitalize()} Algorithm, Language: {language.title()}, Chunk Size: {chunk_size}, Overlap: {overlap_size}){title_suffix}'
    )
    plot_title = title or default_title

    x_axis_label = xlabel or group_by_column.replace("_", " ").title()
    y_axis_label = ylabel or "Dataset Success Rate" # Use specific label

    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.ylim(0, 1.05) # Success rate is between 0 and 1

    # Adjust legend
    try:
        current_legend = ax.get_legend()
        if current_legend:
            legend_title = hue_column.replace("_", " ").title()
            current_legend.set_title(legend_title)
            # Optional: Move legend outside if it overlaps
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=legend_title)
    except AttributeError:
        pass # No legend generated

    # Use tight_layout *before* potentially moving legend if bbox_inches='tight' is used in savefig
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dataset success box plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

    plt.close()