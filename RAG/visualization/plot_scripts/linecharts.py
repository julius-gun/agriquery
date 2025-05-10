# visualization/plot_scripts/linecharts.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict

# Set a consistent style for seaborn plots
sns.set_theme(style="whitegrid")

# Define a mapping for metric display names, can be expanded
METRIC_DISPLAY_NAMES = {
    "f1_score": "F1 Score",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
}

def create_zeroshot_noise_level_linechart(
    data: pd.DataFrame,
    output_path: str,
    metric_name: str,
    language: str,
    model_sort_order: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 7),
    palette: Optional[Dict[str, str]] = None, # Allow model-specific colors
    markers: bool = True,
) -> None:
    """
    Creates a line plot showing a specific metric over noise_level
    for zeroshot experiments, with lines for different question_models.

    Args:
        data: Pandas DataFrame pre-filtered for 'zeroshot' retrieval_algorithm,
              a single language, and a single metric_type.
              Expected columns: 'noise_level', 'metric_value', 'question_model'.
        output_path: The full path where the plot image will be saved.
        metric_name: The internal name of the metric (e.g., 'f1_score').
        language: The language for which the plot is being generated (for title).
        model_sort_order: Optional list of model names to define the order of lines (question_model).
                          Models not in this list will be appended alphabetically.
        figsize: Tuple specifying the figure size (width, height) in inches.
        palette: Optional dictionary mapping model names to colors. If None, seaborn's default is used.
        markers: Whether to add markers to the data points on the lines.
    """
    metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())

    if data is None or data.empty:
        print(
            f"Warning: No data provided for {metric_display_name} line chart for language '{language}'. Skipping plot: {output_path}"
        )
        return

    required_cols = ["noise_level", "metric_value", "question_model"]
    if not all(col in data.columns for col in required_cols):
        print(
            f"Error: DataFrame for {metric_display_name} line chart (lang: {language}) is missing one or more required columns: {required_cols}. Found: {data.columns.tolist()}. Skipping plot."
        )
        return

    # Create a working copy to avoid SettingWithCopyWarning
    plot_data = data.copy()

    # Ensure noise_level is numeric and sort it for the x-axis order
    plot_data["noise_level"] = pd.to_numeric(plot_data["noise_level"], errors="coerce")
    plot_data.dropna(subset=["noise_level", "metric_value"], inplace=True)

    if plot_data.empty:
        print(
            f"Warning: Data became empty after converting 'noise_level' to numeric or dropping NaNs for {metric_display_name} (lang: {language}). Skipping plot: {output_path}"
        )
        return

    # Determine order for models (hue)
    hue_order = None
    present_models = sorted(plot_data["question_model"].unique())

    if model_sort_order:
        hue_order = [m for m in model_sort_order if m in present_models]
        remaining_models = sorted([m for m in present_models if m not in hue_order])
        hue_order.extend(remaining_models)
    else:
        hue_order = present_models # Default to alphabetical if no sort order provided

    if not hue_order:
        print(
            f"Warning: No models found in data for {metric_display_name} (lang: {language}) after processing. Skipping plot: {output_path}"
        )
        return

    try:
        plt.figure(figsize=figsize)
        
        ax = sns.lineplot(
            x="noise_level",
            y="metric_value",
            hue="question_model",
            hue_order=hue_order,
            data=plot_data,
            palette=palette if palette else sns.color_palette(n_colors=len(hue_order)), # Use provided palette or generate one
            marker="o" if markers else None,
            markersize=8,
            linewidth=2.5,
        )

        ax.set_xlabel("Noise Level (Number of Tokens)", fontsize=12)
        ax.set_ylabel(metric_display_name, fontsize=12)
        ax.set_title(
            f"Zero-shot Performance: {metric_display_name} vs. Noise Level\nLanguage: {language.title()}",
            fontsize=14,
            pad=20
        )
        
        # Improve legend
        handles, labels = ax.get_legend_handles_labels()
        # Truncate long model names in legend
        truncated_labels = []
        for label in labels:
            if len(label) > 35: # Arbitrary length limit for legend items
                truncated_labels.append(label[:32] + "...")
            else:
                truncated_labels.append(label)
        
        ax.legend(handles, truncated_labels, title="Question Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


        # Customize grid and ticks
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xticks(sorted(plot_data["noise_level"].unique())) # Ensure all noise levels are ticks
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.05) # Assuming scores are between 0 and 1

        plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust layout to make space for legend

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Line chart saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error generating line chart for {output_path}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all') # Close all figures to free memory


if __name__ == "__main__":
    print("--- Testing Zero-shot Noise Level Line Chart Creation ---")

    # Create dummy data
    example_data_list = []
    models = ["model_A_very_long_name_for_testing_legend_truncation_feature", "model_B_short", "model_C"]
    noise_levels = [1000, 5000, 10000, 30000, 59000]
    current_language = "german"
    current_metric = "f1_score"

    for model_idx, model in enumerate(models):
        for nl_idx, nl in enumerate(noise_levels):
            # Simulate some score variation
            score = 0.4 + (model_idx * 0.15) - (nl_idx * 0.03) + (hash(model+str(nl)) % 100 / 600)
            score = max(0, min(1, score)) # Ensure score is between 0 and 1
            example_data_list.append({
                "language": current_language, # This would be filtered out by generator
                "question_model": model,
                "retrieval_algorithm": "zeroshot", # Also filtered by generator
                "noise_level": nl,
                "metric_type": current_metric, # This would be the basis for filtering in generator
                "metric_value": score,
            })
    
    example_df = pd.DataFrame(example_data_list)

    # Define output path for the test
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "linecharts")
    os.makedirs(test_output_dir, exist_ok=True)
    
    test_output_path = os.path.join(test_output_dir, f"test_zeroshot_{current_language}_{current_metric}_vs_noise.png")

    # Test with a specific model order
    test_model_sort_order = ["model_B_short", "model_A_very_long_name_for_testing_legend_truncation_feature", "model_C"]
    
    # Example model-specific palette
    test_palette = {
        "model_A_very_long_name_for_testing_legend_truncation_feature": "blue",
        "model_B_short": "green",
        "model_C": "red"
    }


    print(f"\nGenerating example line chart for: Lang={current_language}, Metric={current_metric}")
    print(f"Data sample (as it would be passed to the function):\n{example_df[['noise_level', 'metric_value', 'question_model']].head().to_string()}")
    print(f"Model sort order: {test_model_sort_order}")
    print(f"Palette: {test_palette}")

    create_zeroshot_noise_level_linechart(
        data=example_df, # The generator would pass pre-filtered data
        output_path=test_output_path,
        metric_name=current_metric,
        language=current_language,
        model_sort_order=test_model_sort_order,
        palette=test_palette,
        figsize=(11,6)
    )

    # Test with no specific model order (should default to alphabetical) and no palette
    test_output_path_no_order = os.path.join(test_output_dir, f"test_zeroshot_{current_language}_{current_metric}_vs_noise_no_order_palette.png")
    print(f"\nGenerating example line chart (no model order, no palette) for: Lang={current_language}, Metric={current_metric}")
    create_zeroshot_noise_level_linechart(
        data=example_df,
        output_path=test_output_path_no_order,
        metric_name=current_metric,
        language=current_language,
        model_sort_order=None,
        palette=None,
        figsize=(11,6)
    )

    # Test with different metric
    example_df_acc = example_df.copy()
    example_df_acc["metric_value"] = example_df_acc["metric_value"] * 0.8 + 0.1 # Slightly different values for accuracy
    current_metric_acc = "accuracy"
    test_output_path_acc = os.path.join(test_output_dir, f"test_zeroshot_{current_language}_{current_metric_acc}_vs_noise.png")
    print(f"\nGenerating example line chart for: Lang={current_language}, Metric={current_metric_acc}")
    create_zeroshot_noise_level_linechart(
        data=example_df_acc,
        output_path=test_output_path_acc,
        metric_name=current_metric_acc, # testing display name lookup
        language=current_language,
        model_sort_order=test_model_sort_order,
        figsize=(11,6)
    )


    # Test with empty data
    empty_df = pd.DataFrame(columns=["noise_level", "metric_value", "question_model"])
    test_output_path_empty = os.path.join(test_output_dir, "test_empty_data.png")
    print("\nTesting with empty DataFrame...")
    create_zeroshot_noise_level_linechart(
        data=empty_df,
        output_path=test_output_path_empty,
        metric_name="f1_score",
        language="english"
    )
    
    # Test with data missing a required column
    missing_col_df = example_df.drop(columns=["metric_value"])
    test_output_path_missing_col = os.path.join(test_output_dir, "test_missing_col_data.png")
    print("\nTesting with DataFrame missing 'metric_value' column...")
    create_zeroshot_noise_level_linechart(
        data=missing_col_df,
        output_path=test_output_path_missing_col,
        metric_name="f1_score",
        language="english"
    )

    print("\n--- Line Chart Creation Test Finished ---")
    print(f"Test plots (if any) are in: {test_output_dir}")