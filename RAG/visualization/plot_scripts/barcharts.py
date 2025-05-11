# visualization/plot_scripts/barcharts.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any

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
    figsize_per_facet: Tuple[float, float] = (6, 5), # Width, Height per facet
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
    
    total_fig_width = figsize_per_facet[0] * num_facets
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
            legend=False, # We will create a custom legend if needed, or rely on hue
            sharex=True, # Models are likely the same across algorithms for comparison
            sharey=True
        )

        # Direct Labeling on bars
        for ax in g.axes.flat:
            for patch in ax.patches:
                height = patch.get_height()
                if pd.notna(height) and height > 0: # Only label valid, positive bars
                    ax.text(
                        patch.get_x() + patch.get_width() / 2.,
                        height + 0.01, # Position label slightly above the bar
                        f"{height:.2f}", # Format to 2 decimal places
                        ha="center",
                        va="bottom",
                        fontsize=bar_label_fontsize,
                        color="black",
                        rotation=0 # Can be 90 if labels overlap
                    )
            ax.set_ylim(0, 1.05) # Metrics are typically 0-1
            ax.set_xlabel("Question Model", fontsize=axis_label_fontsize)
            ax.set_ylabel(metric_display_name, fontsize=axis_label_fontsize)
            ax.tick_params(axis='x', rotation=45, labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=tick_label_fontsize)
            
            # Set facet titles (retrieval algorithm)
            if ax.get_title(): # catplot sets column name as title
                 ax.set_title(ax.get_title().split('=')[-1].strip(), fontsize=axis_label_fontsize + 1, weight='bold')


        # Add a clear, overall title
        g.fig.suptitle(
            f"Model Performance: {metric_display_name} by Language and Retrieval Method",
            fontsize=title_fontsize,
            y=1.03 # Adjust y to make space for suptitle
        )
        
        # Add a single legend for all facets
        handles, labels = [], []
        # Collect handles and labels from the first axis that has them
        for ax in g.axes.flat:
            h, l = ax.get_legend_handles_labels()
            if h: # If an axis has legend items
                # Create unique legend items based on labels
                unique_labels = {}
                for handle, label in zip(h,l):
                    if label not in unique_labels and label in language_order:
                         unique_labels[label] = handle
                # Order them according to language_order
                ordered_handles = [unique_labels[lbl] for lbl in language_order if lbl in unique_labels]
                ordered_labels = [lbl for lbl in language_order if lbl in unique_labels]
                handles.extend(ordered_handles)
                labels.extend(ordered_labels)
                break # Found legend items, no need to check other axes
        
        if handles and labels:
            g.fig.legend(handles, labels, title="Language", loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=axis_label_fontsize-1, title_fontsize=axis_label_fontsize)


        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to make space for suptitle and legend

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error generating bar chart for {output_path}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close("all")


if __name__ == "__main__":
    print("--- Testing Model Performance Bar Chart Creation ---")

    # Create dummy data
    example_data_list = []
    models = [
        "model_A_long_name", "model_B", "model_C_compact"
    ]
    # These names for retrieval_algorithm_display will be set by the generator
    algorithms = ["BM25", "Hybrid", "Full Manual"] 
    languages = LANGUAGE_ORDER
    metric_to_test = "f1_score"

    base_value = 0.5
    for algo_idx, algo in enumerate(algorithms):
        for model_idx, model in enumerate(models):
            for lang_idx, lang in enumerate(languages):
                # Simulate some variation
                value = base_value + \
                        (model_idx * 0.1) - \
                        (lang_idx * 0.05) + \
                        (algo_idx * 0.15) + \
                        ((hash(model + lang + algo) % 100) / 500 - 0.1) # Random noise
                value = max(0, min(1, value)) # Clamp between 0 and 1

                example_data_list.append({
                    "question_model": model,
                    "metric_value": value,
                    "language": lang,
                    "retrieval_algorithm_display": algo, # This column name is important
                    "metric_type": metric_to_test # For filtering in a real scenario
                })

    example_df = pd.DataFrame(example_data_list)
    
    # Filter for the specific metric, as the generator would do
    example_df_metric_specific = example_df[example_df["metric_type"] == metric_to_test]


    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "barcharts")
    os.makedirs(test_output_dir, exist_ok=True)

    test_output_path = os.path.join(
        test_output_dir, f"test_model_perf_{metric_to_test}.png"
    )
    
    test_model_sort_order = models # Use the defined order
    test_algo_display_order = algorithms # Use the defined order

    print(
        f"\nGenerating example bar chart for: Metric={metric_to_test}"
    )
    create_model_performance_barchart(
        data=example_df_metric_specific,
        output_path=test_output_path,
        metric_name=metric_to_test,
        model_sort_order=test_model_sort_order,
        algorithm_display_order=test_algo_display_order,
        # language_order and language_palette use defaults
        figsize_per_facet=(7, 5), # Wider facets for long model names + 3 languages
        bar_label_fontsize=6,
        title_fontsize=18
    )

    # Test with a different metric to ensure display name changes
    metric_to_test_acc = "accuracy"
    example_df_acc = example_df.copy() # Make a copy for modification
    example_df_acc["metric_value"] = example_df_acc["metric_value"].apply(lambda x: max(0, x - 0.1)) # Slightly different values
    example_df_metric_specific_acc = example_df_acc[example_df_acc["metric_type"] == metric_to_test] # Whoops, should be metric_to_test_acc
    # Correcting the filter for the accuracy test:
    example_df_metric_specific_acc = example_df_acc[example_df_acc["metric_type"] == metric_to_test_acc]
    # This will be empty if metric_type was not set to "accuracy" in dummy data generation.
    # Let's adjust dummy data generation slightly for a quick test.
    # For simplicity in this test, we'll just reuse f1_score data but pass "accuracy" as metric_name.
    # In a real scenario, the generator would filter correctly.
    if example_df_metric_specific_acc.empty: # If still empty (due to earlier logic)
        print(f"Warning: No data for {metric_to_test_acc} in dummy set. Reusing F1 data for test plot structure.")
        example_df_metric_specific_acc = example_df_metric_specific.copy() # Use f1 data


    test_output_path_acc = os.path.join(
        test_output_dir, f"test_model_perf_{metric_to_test_acc}.png"
    )
    print(
        f"\nGenerating example bar chart for: Metric={metric_to_test_acc}"
    )
    create_model_performance_barchart(
        data=example_df_metric_specific_acc, # or example_df_metric_specific if accuracy data is problematic
        output_path=test_output_path_acc,
        metric_name=metric_to_test_acc, # This is what matters for title/labels
        model_sort_order=test_model_sort_order,
        algorithm_display_order=test_algo_display_order
    )

    print("\n--- Bar Chart Creation Test Finished ---")
    print(f"Test plots (if any) are in: {test_output_dir}")