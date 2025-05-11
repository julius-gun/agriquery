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
    algorithm_display_order: Optional[List[str]] = None, # For sorting methods within each model group
    language_order: List[str] = LANGUAGE_ORDER,
    language_palette: Dict[str, str] = LANGUAGE_PALETTE,
    figsize: Tuple[float, float] = (20, 10), # Overall figure size
    bar_label_fontsize: int = 7,
    bar_label_rotation: int = 0,
    title_fontsize: int = 16,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int = 10,
    xtick_label_rotation: int = 45,
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 12
) -> None:
    """
    Creates a single grouped bar chart showing a specific metric for different models and methods.
    Each model-method combination is a group on the x-axis, with languages as bars within that group.

    Args:
        data: Pandas DataFrame. Expected columns: 'question_model', 'metric_value', 'language',
              'retrieval_algorithm_display'.
        output_path: The full path where the plot image will be saved.
        metric_name: The internal name of the metric (e.g., 'f1_score').
        model_sort_order: Order for models on the x-axis.
        algorithm_display_order: Order for retrieval algorithms within each model group.
        language_order: Order for languages (hue).
        language_palette: Color mapping for languages.
        figsize: Tuple (width, height) for the entire plot.
        bar_label_fontsize: Font size for direct labels on bars.
        bar_label_rotation: Rotation for bar labels.
        title_fontsize: Font size for the main plot title.
        axis_label_fontsize: Font size for axis labels.
        tick_label_fontsize: Font size for tick labels.
        xtick_label_rotation: Rotation for x-axis tick labels (model|method names).
        legend_fontsize: Font size for legend text.
        legend_title_fontsize: Font size for legend title.
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
    plot_data.dropna(subset=["metric_value"], inplace=True) # Keep rows with metric_value=0

    if plot_data.empty:
        print(
            f"Warning: Data became empty after type conversion or NaN drop for {metric_display_name}. Skipping plot: {output_path}"
        )
        return

    # Create a composite x-axis category: Model | Method
    plot_data['x_category'] = plot_data['question_model'] + " | " + plot_data['retrieval_algorithm_display']

    # Determine the order of x_categories
    x_category_order = []
    # Use unique values from data as default if sort orders are not provided
    effective_model_sort_order = model_sort_order if model_sort_order is not None else plot_data['question_model'].unique().tolist()
    effective_algorithm_display_order = algorithm_display_order if algorithm_display_order is not None else plot_data['retrieval_algorithm_display'].unique().tolist()

    present_categories = set(plot_data['x_category'].unique())

    for model in effective_model_sort_order:
        for algo in effective_algorithm_display_order:
            category = f"{model} | {algo}"
            if category in present_categories:
                x_category_order.append(category)
    
    # Add any categories present in data but not formed by the sort orders (e.g. if sort orders are incomplete)
    # This ensures all data is plotted, respecting the initial sort as much as possible.
    missing_categories = sorted(list(present_categories - set(x_category_order))) # Sort for some consistency
    x_category_order.extend(missing_categories)


    if not x_category_order:
        print(f"Warning: No x-axis categories could be determined for {metric_display_name}. Skipping plot.")
        return

    current_palette = language_palette.copy()
    current_language_order = language_order[:] # Make a copy
    for lang in plot_data['language'].unique():
        if lang not in current_palette:
            current_palette[lang] = DEFAULT_LANGUAGE_COLOR
            if lang not in current_language_order:
                current_language_order.append(lang)
    
    try:
        g = sns.catplot(
            x="x_category",
            y="metric_value",
            hue="language",
            data=plot_data,
            kind="bar",
            order=x_category_order,
            hue_order=current_language_order,
            palette=current_palette,
            height=figsize[1],
            aspect=figsize[0] / figsize[1],
            legend=False, # Will add custom legend
            sharey=True # Y-axis should be shared (0-1 scale)
        )

        ax = g.ax # The single Axes object

        # Direct Labeling on bars
        for patch in ax.patches:
            height = patch.get_height()
            if pd.notna(height): # Show label for all valid heights, including 0
                ax.text(
                    patch.get_x() + patch.get_width() / 2.,
                    height + 0.01 if height > 0 else 0.01, # Position label slightly above bar, or at 0.01 for zero-height bars
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=bar_label_fontsize,
                    color="black",
                    rotation=bar_label_rotation
                )
        
        ax.set_ylim(0, 1.05) # Metrics are typically 0-1
        ax.set_xlabel("Model | Retrieval Method", fontsize=axis_label_fontsize)
        ax.set_ylabel(metric_display_name, fontsize=axis_label_fontsize)
        
        # Set x-tick labels and rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_label_rotation, fontsize=tick_label_fontsize)
        if xtick_label_rotation > 0 and xtick_label_rotation < 90:
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
        elif xtick_label_rotation == 90:
             ax.set_xticklabels(ax.get_xticklabels(), ha='center', rotation_mode='anchor') # Center for 90 deg
        else: # 0 or other, default to center
            ax.set_xticklabels(ax.get_xticklabels(), ha='center')

        ax.tick_params(axis='y', labelsize=tick_label_fontsize)

        # Add a clear, overall title
        g.fig.suptitle(
            f"Model Performance: {metric_display_name} by Language and Retrieval Method",
            fontsize=title_fontsize,
            y=1.00 # Adjust y to make space for suptitle, may need tweaking
        )
        
        # Add a single legend for all facets
        handles, labels = ax.get_legend_handles_labels()
        
        if handles and labels:
            # Create unique legend items based on labels, ordered by current_language_order
            unique_labels_map = {}
            for handle, label_text in zip(handles, labels):
                if label_text not in unique_labels_map and label_text in current_language_order:
                     unique_labels_map[label_text] = handle
            
            ordered_handles = [unique_labels_map[lbl] for lbl in current_language_order if lbl in unique_labels_map]
            ordered_labels = [lbl for lbl in current_language_order if lbl in unique_labels_map]

            if ordered_handles and ordered_labels:
                g.fig.legend(
                    ordered_handles,
                    ordered_labels,
                    title="Language",
                    loc='upper right', # Position of the legend box
                    bbox_to_anchor=(0.98, 0.98), # Fine-tune position relative to figure
                    fontsize=legend_fontsize,
                    title_fontsize=legend_title_fontsize,
                    frameon=True,
                    shadow=True
                )

        # Adjust layout to prevent overlap and ensure everything fits
        # The rect might need adjustment if suptitle or legend is large or x-labels are very long.
        # rect=[left, bottom, right, top]
        # Giving more space at the bottom for rotated x-labels, and at top for suptitle/legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])


        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # bbox_inches="tight" is crucial for saving the full figure with rotated labels
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error generating bar chart for {output_path}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(g.fig) # Close the specific figure associated with FacetGrid


if __name__ == "__main__":
    print("--- Testing Model Performance Bar Chart Creation (Single Plot Style) ---")

    # Create dummy data
    example_data_list = []
    # Using longer model names and more methods to test layout
    models = [
        "gemini-2.5-pro-preview-05-06", 
        "super-llama-3000-long-name-variant-alpha", 
        "model_C_compact",
        "another_very_long_model_name_that_needs_space"
    ]
    algorithms = [
        "BM25", 
        "Hybrid (BM25 + Embedding)", 
        "Embedding Retriever",
        "ZeroShot (59k tokens)" # Specific zeroshot variant
    ] 
    languages = LANGUAGE_ORDER # ["english", "french", "german"]
    metric_to_test = "f1_score"

    base_value = 0.3
    for model_idx, model in enumerate(models):
        for algo_idx, algo in enumerate(algorithms):
            # Simulate some algos not being available for all models
            if model == "model_C_compact" and algo == "Embedding Retriever":
                continue
            if model == "gemini-2.5-pro-preview-05-06" and algo == "BM25": # Simulate missing data point
                continue

            for lang_idx, lang in enumerate(languages):
                value = base_value + \
                        (model_idx * 0.05) - \
                        (lang_idx * 0.03) + \
                        (algo_idx * 0.10) + \
                        ((hash(model + lang + algo) % 200) / 1000 - 0.1) # Random noise
                value = max(0, min(1, value)) # Clamp between 0 and 1
                
                # Simulate some zero scores
                if model == "model_C_compact" and algo == "Hybrid (BM25 + Embedding)" and lang == "french":
                    value = 0.0
                if model == "another_very_long_model_name_that_needs_space" and algo == "ZeroShot (59k tokens)" and lang == "german":
                    value = 0.0


                example_data_list.append({
                    "question_model": model,
                    "metric_value": value,
                    "language": lang,
                    "retrieval_algorithm_display": algo,
                    "metric_type": metric_to_test 
                })
    
    # Add a case where a model has only one algorithm
    for lang_idx, lang in enumerate(languages):
        value = 0.6 + (lang_idx * 0.05)
        example_data_list.append({
            "question_model": "special_model_one_algo",
            "metric_value": value,
            "language": lang,
            "retrieval_algorithm_display": "UniqueAlgo",
            "metric_type": metric_to_test
        })
    models.append("special_model_one_algo") # Add to model list for sort order

    example_df = pd.DataFrame(example_data_list)
    
    example_df_metric_specific = example_df[example_df["metric_type"] == metric_to_test]

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "barcharts_single_plot")
    os.makedirs(test_output_dir, exist_ok=True)

    test_output_path = os.path.join(
        test_output_dir, f"test_single_model_method_perf_{metric_to_test}.png"
    )
    
    # Define sort orders (can include items not in data, they will be skipped)
    test_model_sort_order = models # Use the order of definition
    test_algo_display_order = algorithms + ["UniqueAlgo"] # Use order of definition, add the unique one

    # Calculate figsize dynamically based on number of x-categories
    num_x_categories = 0
    present_cats_test = set(example_df_metric_specific['question_model'] + " | " + example_df_metric_specific['retrieval_algorithm_display'])
    for m in test_model_sort_order:
        for a in test_algo_display_order:
            if f"{m} | {a}" in present_cats_test:
                num_x_categories +=1
    
    # Adjust width: base_width + width_per_category * num_categories
    # Adjust height: base_height + height_for_labels_if_rotated
    plot_width = max(15, num_x_categories * 1.0) # 1.0 inch per x-category group (adjust as needed)
    plot_height = 10  # Fixed height, or adjust based on label rotation

    print(
        f"\nGenerating example single bar chart for: Metric={metric_to_test}"
    )
    print(f"Number of x-axis categories: {num_x_categories}")
    print(f"Calculated figsize: ({plot_width}, {plot_height})")

    create_model_performance_barchart(
        data=example_df_metric_specific,
        output_path=test_output_path,
        metric_name=metric_to_test,
        model_sort_order=test_model_sort_order,
        algorithm_display_order=test_algo_display_order,
        figsize=(plot_width, plot_height),
        bar_label_fontsize=6,
        title_fontsize=18,
        xtick_label_rotation=60, # Test with a steeper rotation
        tick_label_fontsize=8 # Smaller tick labels for more categories
    )

    # Test with another metric
    metric_to_test_acc = "accuracy"
    example_df_acc = example_df.copy()
    example_df_acc["metric_value"] = example_df_acc["metric_value"].apply(lambda x: max(0, x * 0.9)) # Slightly different values
    example_df_acc["metric_type"] = metric_to_test_acc # Change metric type for all rows
    
    test_output_path_acc = os.path.join(
        test_output_dir, f"test_single_model_method_perf_{metric_to_test_acc}.png"
    )
    print(
        f"\nGenerating example single bar chart for: Metric={metric_to_test_acc}"
    )
    create_model_performance_barchart(
        data=example_df_acc,
        output_path=test_output_path_acc,
        metric_name=metric_to_test_acc,
        model_sort_order=test_model_sort_order,
        algorithm_display_order=test_algo_display_order,
        figsize=(plot_width, plot_height), # Reuse calculated size
        bar_label_fontsize=6,
        xtick_label_rotation=75, # Test different rotation
    )

    print("\n--- Single Bar Chart Creation Test Finished ---")
    print(f"Test plots (if any) are in: {test_output_dir}")