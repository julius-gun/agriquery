# visualization/plot_scripts/barcharts.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import numpy as np # Import numpy for axes checking

# Set a consistent style for seaborn plots
sns.set_theme(style="whitegrid")

# Define a mapping for metric display names
METRIC_DISPLAY_NAMES = {
    "f1_score": "F1 Score",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
}

# Define consistent ordering and colors for languages
# These can be expanded or loaded from config if needed
LANGUAGE_ORDER = ["english", "french", "german"]
LANGUAGE_PALETTE = {
    "english": "#1f77b4",  # Muted blue
    "french": "#ff7f0e",   # Safety orange
    "german": "#2ca02c"    # Cooked asparagus green
}
# A fallback color for any unexpected languages
DEFAULT_LANGUAGE_COLOR = "#7f7f7f" # Medium gray


def create_model_performance_barchart(
    data: pd.DataFrame,
    output_path: str,
    metric_name: str,
    model_sort_order: Optional[List[str]] = None,
    algorithm_display_order: Optional[List[str]] = None,
    language_order: List[str] = LANGUAGE_ORDER,
    language_palette: Dict[str, str] = LANGUAGE_PALETTE,
    figsize_per_facet: Tuple[float, float] = (7.5, 5.5), # Width, Height per facet - increased default width
    bar_label_fontsize: int = 7,
    title_fontsize: int = 16,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int = 10
) -> None:
    """
    Creates a grouped bar chart showing a specific metric for different models,
    grouped by language, and faceted by retrieval algorithm.

    Args:
        data: Pandas DataFrame pre-filtered for a single metric_type.
              Expected columns: 'question_model', 'metric_value', 'language',
                                'retrieval_algorithm_display'.
        output_path: The full path where the plot image will be saved.
        metric_name: The internal name of the metric (e.g., 'f1_score').
        model_sort_order: Order for models on the x-axis.
        algorithm_display_order: Order for algorithm facets (columns).
        language_order: Order for languages (hue).
        language_palette: Color mapping for languages.
        figsize_per_facet: Tuple (width, height) for each facet in the plot.
                           Increasing width can help fit long model names on x-axis.
        bar_label_fontsize: Font size for direct labels on bars.
        title_fontsize: Font size for the main plot title.
        axis_label_fontsize: Font size for axis labels.
        tick_label_fontsize: Font size for tick labels.
    """
    metric_display_name = METRIC_DISPLAY_NAMES.get(
        metric_name, metric_name.replace("_", " ").title()
    )

    if data is None or data.empty:
        print(
            f"Warning: No data provided for {metric_display_name} bar chart. Skipping plot: {output_path}"
        )
        return

    required_cols = [
        "question_model", "metric_value", "language", "retrieval_algorithm_display"
    ]
    if not all(col in data.columns for col in required_cols):
        print(
            f"Error: DataFrame for {metric_display_name} bar chart is missing one or more required columns: {required_cols}. Found: {data.columns.tolist()}. Skipping plot."
        )
        return

    plot_data = data.copy()
    plot_data["metric_value"] = pd.to_numeric(plot_data["metric_value"], errors="coerce")
    # Do not drop rows with metric_value == 0 here if we want to plot them.
    # Drop only if metric_value is NaN (due to conversion error, for example).
    plot_data.dropna(subset=["metric_value"], inplace=True)


    if plot_data.empty:
        print(
            f"Warning: Data became empty after type conversion or NaN drop for {metric_display_name}. Skipping plot: {output_path}"
        )
        return

    # Ensure all languages in data have a color, use default if not in palette
    current_palette = language_palette.copy()
    for lang in plot_data['language'].unique():
        if lang not in current_palette:
            current_palette[lang] = DEFAULT_LANGUAGE_COLOR
            if lang not in language_order: # Add to order if new
                language_order = language_order + [lang]


    # Determine the number of facets for figure size calculation
    num_facets = plot_data['retrieval_algorithm_display'].nunique()
    if num_facets == 0:
        print(f"Warning: No retrieval algorithms found in data for {metric_display_name}. Skipping plot.")
        return
    
    # total_fig_width is handled by catplot's height and aspect calculation
    fig_height = figsize_per_facet[1]

    try:
        # Using catplot for faceting
        g = sns.catplot(
            x="question_model",
            y="metric_value",
            hue="language",
            col="retrieval_algorithm_display",
            data=plot_data,
            kind="bar",
            order=model_sort_order,
            hue_order=language_order,
            palette=current_palette,
            col_order=algorithm_display_order,
            height=fig_height,
            aspect=figsize_per_facet[0] / fig_height, # aspect = width / height
            legend=False, 
            sharex=True, 
            sharey=True
        )

        # Determine the number of columns in the facet grid
        # g.axes is a numpy array of AxesSubplot objects.
        # If it's 1D (e.g., col_wrap=1 or only one col specified), shape will be (N,)
        # If it's 2D, shape will be (N_rows, N_cols)
        if g.axes.ndim == 1:
            num_facet_cols = len(g.axes) # Or 1 if only one column truly
            if num_facets > 1 and (algorithm_display_order and len(algorithm_display_order) >1 ): # if multiple facets are plotted in a single row
                 num_facet_cols = len(g.axes)
            else: # if only one facet or explicitly col_wrap=1 (not used by catplot directly with 'col')
                 num_facet_cols = 1

        else: # g.axes.ndim == 2
            num_facet_cols = g.axes.shape[1]


        for ax_idx, ax in enumerate(g.axes.flat):
            y_min_ax, y_max_ax = ax.get_ylim()
            # Use a small percentage of the y-axis range as offset for labels
            # to make it adaptive to different y-axis scales if sharey=False was used.
            # With sharey=True, this will be consistent across facets.
            label_y_offset = (y_max_ax - y_min_ax) * 0.015 

            for patch in ax.patches:
                height = patch.get_height()
                # Show label if height is not NaN (includes 0)
                if pd.notna(height): 
                    ax.text(
                        patch.get_x() + patch.get_width() / 2.,
                        height + label_y_offset, # Position label slightly above the bar
                        f"{height:.2f}", # Format to 2 decimal places
                        ha="center",
                        va="bottom", # Anchors bottom of text to the y-position
                        fontsize=bar_label_fontsize,
                        color="black",
                        rotation=0 # Default horizontal. Can be set to 90 for very crowded bars.
                    )
            
            # Set y-axis limits, ensuring space for labels, especially near 1.0
            # Metrics are typically 0-1. Giving a bit more room at the top.
            ax.set_ylim(0, 1.1) 
            ax.set_xlabel("Question Model", fontsize=axis_label_fontsize)
            
            # Only set y-axis label for the first facet in each row
            # This is equivalent to checking if ax_idx is a multiple of num_facet_cols
            if ax_idx % num_facet_cols == 0:
                 ax.set_ylabel(metric_display_name, fontsize=axis_label_fontsize)
            else:
                 ax.set_ylabel("")

            # Apply rotation and font size to x-tick labels
            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            # Set rotation and horizontal alignment for x-tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            ax.tick_params(axis='y', labelsize=tick_label_fontsize)
            
            if ax.get_title(): 
                 ax.set_title(ax.get_title().split('=')[-1].strip(), fontsize=axis_label_fontsize + 1, weight='bold')


        g.fig.suptitle(
            f"Model Performance: {metric_display_name} by Language and Retrieval Method",
            fontsize=title_fontsize,
            y=1.03 
        )
        
        handles, labels = [], []
        # ... (legend handling remains the same)
        for ax_loop_idx, ax_leg in enumerate(g.axes.flat): # Renamed ax to ax_leg to avoid conflict
            # Check if this axis is the first one, which typically has all legend items
            # or if it has any legend items at all.
            # With sharey=True and a common hue, any axis should give the same legend items.
            h, l = ax_leg.get_legend_handles_labels()
            if h: 
                unique_labels_dict = {} # Use a dictionary to ensure unique handles by label
                for handle, label_text in zip(h,l): # Renamed label to label_text
                    # Only add if the label is in our defined language_order and not already added
                    if label_text not in unique_labels_dict and label_text in language_order:
                         unique_labels_dict[label_text] = handle
                
                # Order handles and labels according to language_order
                ordered_handles = [unique_labels_dict[lbl] for lbl in language_order if lbl in unique_labels_dict]
                ordered_labels = [lbl for lbl in language_order if lbl in unique_labels_dict]
                
                # Check if we actually got any valid legend items after filtering
                if ordered_handles and ordered_labels:
                    handles.extend(ordered_handles)
                    labels.extend(ordered_labels)
                    break # Found legend items, no need to check other axes
        
        if handles and labels:
            # Position legend carefully, e.g., to the right of the plot area
            # or adjust bbox_to_anchor if it overlaps with title or plot.
            # Example: Place legend outside, to the right. Adjust figure right margin if needed.
            g.fig.legend(
                handles, labels, 
                title="Language", 
                loc='center right', # More robust positioning outside plot
                bbox_to_anchor=(1.05, 0.5), # Adjust x to move further right, y for vertical center
                fontsize=axis_label_fontsize-1, 
                title_fontsize=axis_label_fontsize
            )


        # Adjust layout to make space for suptitle, legend, and rotated x-labels
        # rect might need adjustment if legend is outside, or rely more on bbox_inches='tight'
        plt.tight_layout(rect=[0, 0.03, 0.95, 0.97]) # Adjust left/right for legend, bottom for x-labels

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight") # bbox_inches often helps fit everything
        print(f"Bar chart saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error generating bar chart for {output_path}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close("all")


if __name__ == "__main__":
    print("--- Testing Model Performance Bar Chart Creation ---")

    example_data_list = []
    models = [
        "gemini-1.5-pro-preview-05-14", "model_B_medium_name", "model_C_short",
        "another_very_long_model_name_that_tests_wrapping_and_spacing_for_x_axis"
    ]
    algorithms = ["Hybrid RAG", "Full Manual 59k tokens", "Embedding RAG"] # Added one more for 3 facets
    languages = LANGUAGE_ORDER
    metric_to_test = "f1_score"

    base_value = 0.5
    for algo_idx, algo in enumerate(algorithms):
        for model_idx, model in enumerate(models):
            for lang_idx, lang in enumerate(languages):
                value = base_value + \
                        (model_idx * 0.05) - \
                        (lang_idx * 0.1) + \
                        (algo_idx * 0.2) + \
                        ((hash(model + lang + algo) % 100) / 500 - 0.1) 
                value = max(0, min(1, value)) 
                # Include a zero value for testing
                if model == models[0] and lang == languages[0] and algo == algorithms[0]:
                    value = 0.0

                example_data_list.append({
                    "question_model": model,
                    "metric_value": value,
                    "language": lang,
                    "retrieval_algorithm_display": algo, 
                    "metric_type": metric_to_test 
                })

    example_df = pd.DataFrame(example_data_list)
    
    example_df_metric_specific = example_df[example_df["metric_type"] == metric_to_test]

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "barcharts")
    os.makedirs(test_output_dir, exist_ok=True)

    test_output_path = os.path.join(
        test_output_dir, f"test_model_perf_{metric_to_test}_v4.png" # Incremented version
    )
    
    test_model_sort_order = models 
    test_algo_display_order = algorithms

    print(
        f"\nGenerating example bar chart for: Metric={metric_to_test}"
    )
    create_model_performance_barchart(
        data=example_df_metric_specific,
        output_path=test_output_path,
        metric_name=metric_to_test,
        model_sort_order=test_model_sort_order,
        algorithm_display_order=test_algo_display_order,
        figsize_per_facet=(8, 6), # Adjusted for potentially longer labels
        bar_label_fontsize=7,
        title_fontsize=16 
    )
    # ... (rest of the test cases from the original file if needed) ...
    print("\n--- Bar Chart Creation Test Finished ---")
    print(f"Test plots (if any) are in: {test_output_dir}")