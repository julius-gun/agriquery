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
    palette: Optional[Dict[str, str]] = None,
    markers: bool = True,
    label_fontsize: int = 9,
    label_x_offset_factor: float = 0.015,  # Percentage of x-axis range for label offset
    label_min_y_sep_factor: float = 0.04,  # Percentage of y-axis range for min vertical separation
) -> None:
    """
    Creates a line plot showing a specific metric over noise_level
    for zeroshot experiments, with direct line labeling.

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
        label_fontsize: Font size for direct labels.
        label_x_offset_factor: Factor for x-offset of labels from line ends.
        label_min_y_sep_factor: Factor for minimum y-separation between labels.
    """
    metric_display_name = METRIC_DISPLAY_NAMES.get(
        metric_name, metric_name.replace("_", " ").title()
    )

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
        hue_order = present_models  # Default to alphabetical if no sort order provided

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
            palette=palette
            if palette
            else sns.color_palette(
                n_colors=len(hue_order)
            ),  # Use provided palette or generate one
            marker="o" if markers else None,
            markersize=8,
            linewidth=2.5,
            legend=False,  # Explicitly disable legend in lineplot call
        )

        ax.set_xlabel("Context (Number of Tokens)", fontsize=12)
        ax.set_ylabel(metric_display_name, fontsize=12)
        ax.set_title(
            f"In-Context Performance: {metric_display_name} vs. Number of Tokens\nLanguage: {language.title()}",
            fontsize=14,
            pad=20,
        )

        # # Improve legend
        # handles, labels = ax.get_legend_handles_labels()
        # # Truncate long model names in legend
        # truncated_labels = []
        # for label in labels:
        #     if len(label) > 35: # Arbitrary length limit for legend items
        #         truncated_labels.append(label[:32] + "...")
        #     else:
        #         truncated_labels.append(label)

        # ax.legend(handles, truncated_labels, title="Question Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Customize grid and ticks
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        # Ensure all unique noise levels are ticks if they are not too many, otherwise auto
        unique_noise_levels = sorted(plot_data["noise_level"].unique())
        if len(unique_noise_levels) <= 10:  # Arbitrary limit for explicit ticks
            plt.xticks(unique_noise_levels)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)

        # --- Direct Labeling ---
        lines = ax.get_lines()
        model_to_color = {
            hue_order[i]: lines[i].get_color()
            for i in range(len(hue_order))
            if i < len(lines)
        }

        labels_to_draw = []
        for model_name in hue_order:
            model_data = plot_data[
                plot_data["question_model"] == model_name
            ].sort_values(by="noise_level")
            if not model_data.empty:
                last_point = model_data.iloc[-1]
                labels_to_draw.append(
                    {
                        "text": model_name,
                        "x": last_point["noise_level"],
                        "y": last_point["metric_value"],
                        "color": model_to_color.get(model_name, "black"),
                    }
                )

        labels_to_draw.sort(key=lambda L: L["y"])  # Sort by y-value for adjustment

        plot_ymin, plot_ymax = ax.get_ylim()
        effective_min_text_sep_y = label_min_y_sep_factor * (plot_ymax - plot_ymin)
        last_adjusted_y = -float(
            "inf"
        )  # y-coordinate of the previously placed label's center

        adjusted_labels_info = []  # Store all info needed for drawing, including adjusted y

        for i in range(len(labels_to_draw)):
            label_info = labels_to_draw[i]
            target_y = label_info["y"]

            # Adjust if current label's target y is too close to the previous one's adjusted y
            adjusted_y = max(target_y, last_adjusted_y + effective_min_text_sep_y)

            # Clamp to plot boundaries, ensuring space for half the text height (approx.)
            # This avoids the text center being right at the edge.
            half_sep = effective_min_text_sep_y / 2
            adjusted_y = max(plot_ymin + half_sep, adjusted_y)
            adjusted_y = min(plot_ymax - half_sep, adjusted_y)

            label_info["final_y"] = adjusted_y
            last_adjusted_y = (
                adjusted_y  # Update with the y-center of the label just placed
            )
            adjusted_labels_info.append(label_info)

        # Draw labels
        plot_xmin, plot_xmax = ax.get_xlim()
        x_range = plot_xmax - plot_xmin

        for label_info in adjusted_labels_info:
            x_pos, y_pos = label_info["x"], label_info["final_y"]

            # Determine horizontal alignment: if line ends in the rightmost 20% of plot, label to its left
            is_at_end = x_pos >= plot_xmax - x_range * 0.2
            ha = "right" if is_at_end else "left"

            # Calculate final x position with offset
            x_offset = x_range * label_x_offset_factor
            final_x = x_pos - x_offset if is_at_end else x_pos + x_offset

            # Truncate long model names for labels
            max_label_len = 30  # Max length for model name in label
            display_text = label_info["text"]
            if len(display_text) > max_label_len:
                display_text = display_text[: max_label_len - 3] + "..."

            ax.text(
                final_x,
                y_pos,
                display_text,
                fontsize=label_fontsize,
                color=label_info["color"],
                verticalalignment="center",
                horizontalalignment=ha,
                bbox=dict(
                    facecolor="white", alpha=0.6, edgecolor="none", pad=0.2
                ),  # Subtle background
            )
        # --- End Direct Labeling ---

        plt.tight_layout()
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
        plt.close("all")  # Close all figures to free memory


if __name__ == "__main__":
    print("--- Testing Zero-shot Noise Level Line Chart Creation ---")

    # Create dummy data
    example_data_list = []
    # More models to better test labeling and overlap
    models = [
        "model_A_very_long_name_for_testing_label_truncation_and_overlap_feature",
        "model_B_short",
        "model_C_medium_length",
        "model_D_another_one",
        "model_E_close_to_D",
        "model_F_highest_performer",
    ]
    noise_levels = [
        1000,
        5000,
        10000,
        20000,
        30000,
        40000,
        50000,
        59000,
    ]  # More noise levels
    current_language = "german"
    current_metric = "f1_score"

    for model_idx, model in enumerate(models):
        base_score = 0.25 + (model_idx * 0.08)  # Spread out base scores
        for nl_idx, nl in enumerate(noise_levels):
            # Simulate score decay with noise, and some model-specific variation
            score = base_score - (nl_idx * 0.015) + (hash(model + str(nl)) % 100 / 800)
            # Intentional overlaps for testing
            if model == "model_D_another_one":
                score += 0.02
            if model == "model_E_close_to_D":
                score -= 0.01  # model_E should be close to model_D
            if model == "model_F_highest_performer":
                score += 0.1  # Make F distinctly higher

            score = max(0, min(1, score))
            example_data_list.append(
                {
                    "language": current_language,  # This would be filtered out by generator
                    "question_model": model,
                    "retrieval_algorithm": "zeroshot",  # Also filtered by generator
                    "noise_level": nl,
                    "metric_type": current_metric,  # This would be the basis for filtering in generator
                    "metric_value": score,
                }
            )

    example_df = pd.DataFrame(example_data_list)

    # Define output path for the test
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "linecharts")
    os.makedirs(test_output_dir, exist_ok=True)

    test_output_path = os.path.join(
        test_output_dir, f"test_directlabel_{current_language}_{current_metric}.png"
    )
    # Define a sort order; models not in this list will be appended alphabetically
    test_model_sort_order = [
        "model_F_highest_performer",
        "model_A_very_long_name_for_testing_label_truncation_and_overlap_feature",
        "model_D_another_one",
        "model_E_close_to_D",
    ]

    test_palette = {  # Palette for some models
        "model_A_very_long_name_for_testing_label_truncation_and_overlap_feature": "purple",
        "model_B_short": "#FF5733",
        "model_C_medium_length": "#33FF57",
        "model_D_another_one": "#3357FF",
        "model_E_close_to_D": "#FFC300",  # Bright yellow for E
        "model_F_highest_performer": "#00A0A0",  # Teal for F
    }

    print(
        f"\nGenerating example line chart with specific order and palette for: Lang={current_language}, Metric={current_metric}"
    )
    create_zeroshot_noise_level_linechart(
        data=example_df,
        output_path=test_output_path,
        metric_name=current_metric,
        language=current_language,
        model_sort_order=test_model_sort_order,
        palette=test_palette,
        figsize=(14, 8),
        label_fontsize=8,
        label_min_y_sep_factor=0.035,  # Adjusted for potentially denser plot
    )

    # Test with default palette and no specific model order (models will be alphabetical)
    test_output_path_no_order = os.path.join(
        test_output_dir,
        f"test_directlabel_{current_language}_{current_metric}_no_order_palette.png",
    )
    print(
        f"\nGenerating example line chart (alphabetical model order, default palette) for: Lang={current_language}, Metric={current_metric}"
    )
    create_zeroshot_noise_level_linechart(
        data=example_df,
        output_path=test_output_path_no_order,
        metric_name=current_metric,
        language=current_language,
        model_sort_order=None,
        palette=None,  # Defaults
        figsize=(14, 8),
        label_fontsize=8,
        label_min_y_sep_factor=0.035,
    )

    # Test for a different metric (accuracy)
    example_df_acc = example_df.copy()
    # Slightly alter scores for accuracy plot to make it visually different
    example_df_acc["metric_value"] = example_df_acc["metric_value"].apply(
        lambda x: min(1, x * 0.9 + 0.1)
    )
    current_metric_acc = "accuracy"
    test_output_path_acc = os.path.join(
        test_output_dir, f"test_directlabel_{current_language}_{current_metric_acc}.png"
    )
    print(
        f"\nGenerating example line chart for: Lang={current_language}, Metric={current_metric_acc}"
    )
    create_zeroshot_noise_level_linechart(
        data=example_df_acc,
        output_path=test_output_path_acc,
        metric_name=current_metric_acc,
        language=current_language,
        model_sort_order=test_model_sort_order,
        palette=test_palette,
        figsize=(14, 8),
        label_fontsize=8,
        label_min_y_sep_factor=0.035,
    )
    print("\n--- Direct Label Line Chart Creation Test Finished ---")
    print(f"Test plots (if any) are in: {test_output_dir}")
