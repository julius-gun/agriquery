import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict

# Set a consistent style for seaborn plots
sns.set_theme(style="whitegrid")

# Define a mapping for metric display names (can be expanded)
METRIC_DISPLAY_NAMES = {
    "f1_score": "F1 Score",
    "accuracy": "Accuracy",
}

# Define colors for retrieval methods if specific styling is desired (optional)
# sns default palette will be used if this is None or not used.
RETRIEVAL_METHOD_PALETTE = {
    "Hybrid": "#1f77b4",
    "Embedding": "#ff7f0e",
    "Keyword": "#2ca02c",
    "Full Manual": "#d62728",
}

def create_english_retrieval_barchart(
    data: pd.DataFrame,
    output_path: str,
    metric_name: str,
    model_sort_order: Optional[List[str]] = None,
    retrieval_method_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 8),
    bar_label_fontsize: int = 7,
    text_inside_bar_fontsize: int = 14,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 13) -> None:
    """
    Creates a grouped bar chart for English results for a specific metric,
    showing models on the x-axis and retrieval methods as grouped bars.

    Args:
        data: Pandas DataFrame pre-filtered for English language and a single metric_type.
              Expected columns: 'question_model', 'metric_value', 'retrieval_method_display'.
        output_path: The full path where the plot image will be saved.
        metric_name: The internal name of the metric (e.g., 'f1_score', 'accuracy').
        model_sort_order: Order for models on the x-axis.
        retrieval_method_order: Order for retrieval methods (hue).
        figsize: Figure size.
        bar_label_fontsize: Font size for metric values above bars.
        text_inside_bar_fontsize: Font size for retrieval method names inside bars.
        title_fontsize: Font size for the main plot title.
        axis_label_fontsize: Font size for axis labels.
        tick_label_fontsize: Font size for tick labels.
    """
    metric_display_name = METRIC_DISPLAY_NAMES.get(
        metric_name, metric_name.replace("_", " ").title()
    )

    if data is None or data.empty:
        print(
            f"Warning: No data provided for English {metric_display_name} barchart. Skipping plot: {output_path}"
        )
        return

    required_cols = ["question_model", "metric_value", "retrieval_method_display"]
    if not all(col in data.columns for col in required_cols):
        print(
            f"Error: DataFrame for English {metric_display_name} barchart is missing required columns. "
            f"Expected: {required_cols}. Found: {data.columns.tolist()}. Skipping plot."
        )
        return

    plot_data = data.copy()
    plot_data["metric_value"] = pd.to_numeric(plot_data["metric_value"], errors="coerce")
    plot_data.dropna(subset=["metric_value"], inplace=True)

    if plot_data.empty:
        print(
            f"Warning: Data became empty after type conversion or NaN drop for English {metric_display_name}. Skipping plot."
        )
        return

    # Determine unique models and retrieval methods actually present in the data
    # This helps in case model_sort_order or retrieval_method_order contains items not in data
    actual_models = plot_data['question_model'].unique()
    if model_sort_order:
        model_order_for_plot = [m for m in model_sort_order if m in actual_models]
    else:
        model_order_for_plot = sorted(actual_models)

    actual_retrieval_methods = plot_data['retrieval_method_display'].unique()
    if retrieval_method_order:
        retrieval_method_order_for_plot = [r for r in retrieval_method_order if r in actual_retrieval_methods]
    else:
        retrieval_method_order_for_plot = sorted(actual_retrieval_methods)

    if not model_order_for_plot or not retrieval_method_order_for_plot:
        print(f"Warning: No models or retrieval methods to plot for English {metric_display_name} after filtering. Skipping.")
        return

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        x="question_model",
        y="metric_value",
        hue="retrieval_method_display",
        data=plot_data,
        order=model_order_for_plot,
        hue_order=retrieval_method_order_for_plot,
        palette=RETRIEVAL_METHOD_PALETTE if retrieval_method_order_for_plot else None, # Optional: use custom palette
        dodge=True # Ensure bars are dodged
    )

    # Get legend handles and labels directly from the Axes object
    # This should be done before the legend is potentially removed.
    handles, actual_legend_labels_for_hue = [], []
    if ax.get_legend() is not None: # Check if a legend exists
        handles, actual_legend_labels_for_hue = ax.get_legend_handles_labels()


    y_axis_max = plot_data["metric_value"].max()
    label_offset = y_axis_max * 0.01 if pd.notna(y_axis_max) and y_axis_max > 0 else 0.01

    if hasattr(ax, 'containers') and ax.containers:
        # Ensure we have labels to match the containers.
        # The number of containers should match the number of hue categories plotted.
        if len(ax.containers) == len(actual_legend_labels_for_hue):
            for i, container in enumerate(ax.containers):
                current_retrieval_method_label = actual_legend_labels_for_hue[i]

                for patch in container.patches: 
                    height = patch.get_height()
                    width = patch.get_width()
                    x_center = patch.get_x() + width / 2.0

                    if pd.notna(height): 
                        ax.text(
                            x_center,
                            height + label_offset,
                            f"{height:.3f}", 
                            ha="center",
                            va="bottom",
                            fontsize=bar_label_fontsize,
                            color="black"
                        )

                        if height > 0.25: 
                            ax.text(
                                x_center,
                                height / 2.0, 
                                current_retrieval_method_label, 
                                ha="center",
                                va="center",
                                fontsize=text_inside_bar_fontsize,
                                color="white", 
                                rotation=90,
                                weight='bold'
                            )
        elif ax.containers and not actual_legend_labels_for_hue and len(ax.containers) > 0:
             # This case can happen if `hue` was used but for some reason legend labels weren't retrievable
             # or if `hue_order` was specified but no legend was explicitly generated to be captured.
             # Fallback: try to use retrieval_method_order_for_plot if lengths match.
             # This is closer to the original logic but as a fallback.
            print(f"Warning: Could not get legend labels, but bar containers exist for plot: {output_path}. "
                  f"Number of containers: {len(ax.containers)}. "
                  f"Attempting to use `retrieval_method_order_for_plot` (length: {len(retrieval_method_order_for_plot)}) for in-bar labels.")
            if len(ax.containers) == len(retrieval_method_order_for_plot):
                for i, container in enumerate(ax.containers):
                    current_retrieval_method_label = retrieval_method_order_for_plot[i] # Fallback label
                    for patch in container.patches:
                        height = patch.get_height()
                        width = patch.get_width()
                        x_center = patch.get_x() + width / 2.0
                        if pd.notna(height) and height > 0 and height > 0.01:
                             ax.text(
                                x_center, height / 2.0, current_retrieval_method_label,
                                ha="center", va="center", fontsize=text_inside_bar_fontsize,
                                color="white", rotation=90, weight='bold'
                            )
                        # Note: Metric value labels are still added in the primary loop above if this fallback is hit.
                        # We only add the text inside bars here as a fallback.
            else:
                 print(f"Error: Mismatch between number of bar containers ({len(ax.containers)}) "
                      f"and `retrieval_method_order_for_plot` ({len(retrieval_method_order_for_plot)}). "
                      f"Cannot reliably determine in-bar text labels for retrieval methods for plot: {output_path}")

        elif len(ax.containers) != len(actual_legend_labels_for_hue) and (ax.containers or actual_legend_labels_for_hue):
            # General mismatch if both had items but lengths differed.
            print(f"Warning: Mismatch between number of bar containers ({len(ax.containers)}) "
                  f"and retrieved legend labels ({len(actual_legend_labels_for_hue)}). "
                  f"Skipping in-bar text labels for retrieval methods for plot: {output_path}")
        # If both ax.containers and actual_legend_labels_for_hue are empty, no warning needed.
    
    ax.set_xlabel("Model", fontsize=axis_label_fontsize)
    ax.set_ylabel(metric_display_name, fontsize=axis_label_fontsize)
    ax.set_title(
        f"English Language: {metric_display_name} Comparison by Retrieval Method",
        fontsize=title_fontsize,
        pad=20
    )

    ax.tick_params(axis='x', labelsize=tick_label_fontsize) # Set labelsize for x-axis ticks
    # Correctly set rotation and horizontal alignment for x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.tick_params(axis='y', labelsize=tick_label_fontsize) # Set labelsize for y-axis ticks


    # Y-axis limit
    max_val = plot_data["metric_value"].max()
    # Ensure there's always some upper limit even if max_val is 0 or NaN
    upper_y_limit = 1.0 
    if pd.notna(max_val):
        if max_val > 0:
            upper_y_limit = max_val * 1.15 # Add some padding at the top
        else: # max_val is 0 or negative (though metrics are usually >=0)
            upper_y_limit = 0.1 # Small padding if max is 0
    ax.set_ylim(0, upper_y_limit)

    # Remove legend after potentially using its labels
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.tight_layout()

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Barchart saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error generating barchart for {output_path}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close()

if __name__ == "__main__":
    print("--- Testing English Retrieval Barchart Creation ---")

    # Create dummy data for testing
    models_test = ["Model A", "Model B", "Model Long Name C-128k"]
    retrieval_methods_test = ["Hybrid", "Embedding", "Keyword", "Full Manual 59k tokens"]
    
    test_data_list = []
    for model in models_test:
        for i, method in enumerate(retrieval_methods_test):
            # Simulate varying performance
            value = 0.5 + (models_test.index(model) * 0.1) - (i * 0.05) + (hash(model+method) % 100 / 500.0)
            value = max(0, min(1, value)) # Ensure value is between 0 and 1
            if method == "Keyword" and model == "Model A": value = 0.005 # Test very small bar
            test_data_list.append({
                "question_model": model,
                "metric_value": value,
                "retrieval_method_display": method,
                "language": "english", # For context, not directly used by this plot
                "metric_type": "accuracy" # For context
            })
    
    test_df = pd.DataFrame(test_data_list)

    # Create a temporary directory for test plots
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "english_barcharts")
    os.makedirs(test_output_dir, exist_ok=True)
    
    test_output_path = os.path.join(test_output_dir, "test_english_accuracy_barchart.png")

    create_english_retrieval_barchart(
        data=test_df,
        output_path=test_output_path,
        metric_name="accuracy",
        model_sort_order=models_test,
        retrieval_method_order=retrieval_methods_test,
        figsize=(12,7) # Adjust for test
    )

    print(f"Test plot saved to: {test_output_path}")
    print("--- English Retrieval Barchart Test Finished ---")