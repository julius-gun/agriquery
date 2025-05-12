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

# Mapping for language full name to short code for labels
LANGUAGE_CODES = {
    "english": "EN",
    "french": "FR",
    "german": "DE"
    # Add other languages here if needed
}
# Fallback if language not in mapping
DEFAULT_LANGUAGE_CODE = "?"


def create_model_performance_barchart(
    data: pd.DataFrame,
    output_path: str,
    metric_name: str,
    model_sort_order: Optional[List[str]] = None,
    algorithm_display_order: Optional[List[str]] = None,
    language_order: List[str] = LANGUAGE_ORDER,
    language_palette: Dict[str, str] = LANGUAGE_PALETTE,
    figsize_per_facet: Tuple[float, float] = (7.5, 5.5), # Width, Height per facet - increased default width
    bar_label_fontsize: int = 6, # Slightly reduced default size for labels
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
    # Also update language_order to include any new languages found in data
    current_palette = language_palette.copy()
    actual_languages_in_data = sorted(plot_data['language'].unique())
    # Filter language_order to only include languages actually present in the data
    current_lang_order = [lang for lang in language_order if lang in actual_languages_in_data]
    # Add any languages from data that were not in the original language_order
    for lang in actual_languages_in_data:
        if lang not in current_lang_order:
            current_lang_order.append(lang)
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
            hue_order=current_lang_order, # Use the potentially updated language order
            palette=current_palette,      # Use the potentially updated palette
            col_order=algorithm_display_order,
            height=fig_height,
            aspect=figsize_per_facet[0] / fig_height, # aspect = width / height
            legend=False, 
            sharex=True, 
            sharey=True
        )

        # Determine the number of columns in the facet grid
        num_facet_cols = g.axes.shape[1] if g.axes.ndim == 2 else len(g.axes)

        # Get x-tick locations and labels once per plot type
        # These are shared across facets when sharex=True
        # Need to get them from one of the axes, e.g., the first one
        first_ax = g.axes.flat[0]
        # Check if there are any ticks or labels before attempting to access
        x_tick_locations = first_ax.get_xticks() if first_ax.get_xticks() is not None else []
        x_tick_labels = [label.get_text() for label in first_ax.get_xticklabels()] if first_ax.get_xticklabels() is not None else []

        if len(x_tick_locations) != len(x_tick_labels) or not x_tick_locations:
             print(f"Warning: Cannot reliably position labels due to issues with x-ticks. Skipping label drawing.")
             draw_labels = False
        else:
             draw_labels = True


        for ax_idx, ax in enumerate(g.axes.flat):
            # Clear default labels and titles set by seaborn if we are adding custom ones
            # ax.set_title("") # Optionally clear default facet title if using fig.suptitle only
            # ax.set_xlabel("") # Will set custom xlabel later
            # ax.set_ylabel("") # Will set custom ylabel later

            # Get the algorithm display name for the current facet
            facet_algorithm_display = ax.get_title().split('=')[-1].strip()
            # Set title using the extracted value
            ax.set_title(facet_algorithm_display, fontsize=axis_label_fontsize + 1, weight='bold')

            # Use a small percentage of the y-axis range as offset for labels above the bar
            # This makes the offset adaptive if y-axis limits change (though sharey=True here)
            y_min_ax, y_max_ax = ax.get_ylim()
            label_above_offset = (y_max_ax - y_min_ax) * 0.01 # Reduced offset slightly


            if draw_labels:
                # Group patches by the model they belong to based on their x-position
                model_patches_map = {}
                for patch in ax.patches:
                     # Find which x-tick (model) this patch is closest to
                     # Ensure we only consider x-ticks that are actually used/visible on the axis
                     valid_x_tick_locations = [loc for i, loc in enumerate(x_tick_locations) if i < len(x_tick_labels)]
                     if not valid_x_tick_locations: continue # Should not happen if draw_labels is True

                     # Check if the patch's center x-position is within the range of the bars for the first model
                     # This helps filter out potential spurious patches or if axis layout is unexpected
                     patch_center_x = patch.get_x() + patch.get_width()/2.
                     
                     # Find the index of the model this patch belongs to
                     try:
                         # Ensure we only consider x-ticks that are actually used/visible on the axis
                         valid_x_tick_locations = [loc for i, loc in enumerate(x_tick_locations) if i < len(x_tick_labels)]
                         if not valid_x_tick_locations: continue # Should not happen if draw_labels is True

                         closest_model_idx = np.argmin(np.abs(valid_x_tick_locations - patch_center_x))
                         model_name = x_tick_labels[closest_model_idx]
                     except ValueError:
                         # This can happen if valid_x_tick_locations is empty or calculation fails
                         print(f"Warning: Could not find closest model for patch at x={patch_center_x:.2f}. Skipping patch labeling.")
                         continue # Skip labeling for this patch

                     if model_name not in model_patches_map:
                          model_patches_map[model_name] = []
                     model_patches_map[model_name].append(patch)

                # Iterate through the models (x-tick labels in displayed order) and their sorted patches
                for model_name in x_tick_labels: # Iterate models in displayed order on the axis
                     if model_name in model_patches_map:
                          # Sort patches for this model by their x-position to match hue_order
                          # Note: This assumes seaborn orders patches within a group consistently by hue_order.
                          # While generally true, slight variations could occur.
                          sorted_patches = sorted(model_patches_map[model_name], key=lambda p: p.get_x())

                          for hue_index, patch in enumerate(sorted_patches):
                              height = patch.get_height()

                              # --- Label Visibility Threshold ---
                              # Show labels only if height >= 0 (and is not NaN) - MODIFIED CONDITION
                              if pd.notna(height) and height >= 0:
                                   center_x = patch.get_x() + patch.get_width() / 2.

                                   # Add Metric Label (Above the bar, black)
                                   # For height 0 bars, position the label at the bottom of the plot (y=0)
                                   # Or slightly above 0 if you want it *not* touching the x-axis.
                                   # Placing it slightly above 0 might be better for visibility.
                                   # Let's position it slightly above 0 for non-zero values,
                                   # but maybe slightly above 0 for zero values too, or just at 0 + offset?
                                   # Let's keep the same offset logic, but ensure it doesn't go below 0.
                                   metric_text_y = max(0, height) + label_above_offset # Position slightly above the bar or 0
                                   ax.text(
                                       center_x,
                                       metric_text_y,
                                       f"{height:.2f}", # Format to 2 decimal places
                                       ha="center",
                                       va="bottom", # Anchor bottom of text to the y-position
                                       fontsize=bar_label_fontsize,
                                       color="black", # Black text
                                       rotation=0 # Keep horizontal
                                   )

                                   # Add Language Code Label (Inside the bar, white)
                                   # Get language from current_lang_order using the index
                                   try:
                                       lang = current_lang_order[hue_index]
                                       lang_code = LANGUAGE_CODES.get(lang, lang.upper()[:2]) # Fallback code (e.g., "DE" for "german")
                                   except IndexError:
                                       # This can happen if the number of patches doesn't match hue_order length,
                                       # which might occur with empty data points that seaborn doesn't draw patches for.
                                       # In this case, we might not know the language.
                                       lang_code = DEFAULT_LANGUAGE_CODE # Fallback code
                                       lang = "Unknown" # Fallback lang name
                                       print(f"Warning: Could not determine language for patch at x={patch.get_x():.2f}, height={height:.2f}. Using default code.")

                                   # Position inside, vertically centered.
                                   # For height 0, the center is 0, but the text will be 'white' on white, so maybe put it outside?
                                   # The request was for bars that are 0 or a little bit > 0.
                                   # For height 0, maybe place the language code *below* the x-axis or near the metric label?
                                   # Sticking with inside for now, as it works for > 0.
                                   # For height 0, white text inside is invisible. A better approach for 0 bars might be:
                                   # - Metric label: black, slightly above 0 (as done)
                                   # - Language code: white, *only if* height is sufficient to see it inside.
                                   #   Or black, slightly below 0 or near the metric label for height 0.
                                   # Let's stick to "inside" (vertically centered on the bar) for now,
                                   # accepting it's invisible for height 0, but visible for > 0.
                                   # A refinement would be needed for perfect placement on 0 bars.
                                   lang_text_y = height * 0.5 # Position inside, vertically centered
                                   
                                   # Optional: Make language code visible for height 0 bars (requires changing color or position)
                                   # For simplicity based on the request, let's add it where it would be *if* the bar had height > 0,
                                   # which means for height=0, lang_text_y is 0. White text at y=0 won't be visible on a white background.
                                   # We could add an extra check: if height == 0, make color black and position slightly below 0.
                                   lang_label_color = "white"
                                   lang_label_va = "center"
                                   # if height == 0: # Refinement for zero bars - not requested explicitly but improves visibility
                                   #     lang_label_color = "black"
                                   #     lang_text_y = -label_above_offset # Position below axis? Or near metric label?
                                   #     lang_label_va = "top" # Anchor top of text to y-position

                                   ax.text(
                                       center_x,
                                       lang_text_y,
                                       lang_code, # e.g. "EN", "FR"
                                       ha="center",
                                       va=lang_label_va, # Anchor center or top depending on height
                                       fontsize=bar_label_fontsize,
                                       color=lang_label_color, # White text (or black for height 0)
                                       weight='bold' # Make language code bold for readability
                                   )
                              # If height < 0.1, no labels are added due to the 'if' condition.

            # Set y-axis limits, ensuring space for labels above the bar
            # Metrics are typically 0-1. Giving a bit more room at the top based on max height found.
            # Filter plot_data for the current facet BEFORE calculating max height
            facet_plot_data = plot_data[plot_data['retrieval_algorithm_display'] == facet_algorithm_display]
            max_height_in_facet = facet_plot_data['metric_value'].max()

            # Ensure y-limit provides space. Calculate based on max height found, or a default if no data/max is NaN.
            # Add buffer relative to the max height *within this facet*.
            if pd.notna(max_height_in_facet):
                 # Calculate a buffer based on the current y-axis range, scaled by desired label size relative to axis font size
                 # A more robust buffer might be a fixed amount relative to the expected data range (0-1)
                 # Let's add a small fixed amount (e.g., 0.05) plus the calculated offset, ensuring it's at least 1.05
                 # This accounts for space needed above 1.0 if data points reach 1.0
                 buffer_space = label_above_offset + 0.02 # Add the calculated offset plus a small margin
                 upper_ylim = max(1.05, max_height_in_facet + buffer_space)
            else:
                 # If no valid max height (e.g., empty facet, all NaN), use a default limit
                 upper_ylim = 1.05 # Safe default for 0-1 scale

            ax.set_ylim(0, upper_ylim)


            ax.set_xlabel("Question Model", fontsize=axis_label_fontsize)
            
            # Only set y-axis label for the first facet in each row
            # This is equivalent to checking if ax_idx is a multiple of num_facet_cols
            if ax_idx % num_facet_cols == 0:
                 ax.set_ylabel(metric_display_name, fontsize=axis_label_fontsize)
            else:
                 ax.set_ylabel("") # Keep ylabel empty for subsequent columns

            # Apply rotation and font size to x-tick labels
            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            # Set rotation and horizontal alignment for x-tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            ax.tick_params(axis='y', labelsize=tick_label_fontsize)


        # Add main title
        g.fig.suptitle(
            f"Model Performance: {metric_display_name} by Language and Retrieval Method",
            fontsize=title_fontsize,
            y=1.03
        )

        # Handle legend creation based on current_lang_order and current_palette
        handles = []
        labels = []
        # Create dummy patches for the legend handle using the palette colors
        for lang in current_lang_order:
             if lang in current_palette:
                 # Create a simple patch for the legend handle
                 handles.append(plt.matplotlib.patches.Patch(color=current_palette[lang], label=lang))
                 labels.append(lang)

        if handles and labels:
            # Position legend carefully, e.g., to the right of the plot area
            g.fig.legend(
                handles, labels,
                title="Language",
                loc='center right', # Position outside plot
                bbox_to_anchor=(1.02, 0.5), # Adjust x to move further right, y for vertical center
                fontsize=axis_label_fontsize-1,
                title_fontsize=axis_label_fontsize
            )


        # Adjust layout to make space for suptitle, legend, and rotated x-labels
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.97]) # Adjust rect to make space for the legend on the right

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


# --- Model name cleaning function (added for standalone test compatibility) ---
# In a real scenario, this function would likely be in a shared utils file
# and imported by both barchart_generators.py and the __main__ block here.
# Adding it here makes the standalone test run correctly.
def clean_model_name(model_name: str) -> str:
    """
    Applies cleaning rules to a model name for display in standalone test.
    This is a simplified version for testing purposes if not run via main_visualization.
    The primary cleaning happens in barchart_generators.py.
    """
    # This should mirror the logic in barchart_generators.py
    specific_mappings = {
        "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
        "phi3_14B_q4_medium-128k": "phi3 14B",
    }
    cleaned_name = specific_mappings.get(model_name, model_name)

    if cleaned_name == model_name:
        if cleaned_name.endswith("-128k"):
            cleaned_name = cleaned_name.removesuffix("-128k")
        cleaned_name = cleaned_name.replace("_", " ")

    return cleaned_name
# --- End model name cleaning function ---


if __name__ == "__main__":
    print("--- Testing Model Performance Bar Chart Creation ---")

    example_data_list = []
    # Use original names here, will be cleaned before plotting in the test setup
    models = [
        "gemini-1.5-pro-preview-05-14", # No specific map, but general rules apply
        "model_B_medium_name-128k", # General rules apply
        "model_C_short", # No rules apply
        "another_very_long_model_name_that_tests_wrapping_and_spacing_for_x_axis-128k", # General rules apply
        "gemini-2.5-flash-preview-04-17", # Specific map
        "phi3_14B_q4_medium-128k" # Specific map
    ]
    algorithms = ["Hybrid RAG", "Full Manual 59k tokens"] # Reduced to 2 facets for test clarity
    languages = LANGUAGE_ORDER
    metric_to_test = "f1_score"

    base_value = 0.5
    for algo_idx, algo in enumerate(algorithms):
        for model_idx, model_orig in enumerate(models):
            # Clean model name for the *test data* itself to simulate the generator's behavior
            model_cleaned = clean_model_name(model_orig)
            for lang_idx, lang in enumerate(languages):
                value = base_value + \
                        (model_idx * 0.03) - \
                        (lang_idx * 0.08) + \
                        (algo_idx * 0.15) + \
                        ((hash(model_orig + lang + algo) % 100) / 800 - 0.05) # Smaller random variation
                value = max(0, min(1, value))

                # Introduce values < 0.1 and exactly 0 for testing the threshold
                if model_orig == models[0] and lang == languages[0] and algo == algorithms[0]:
                    value = 0.0 # Exactly 0
                if model_orig == models[1] and lang == languages[1] and algo == algorithms[0]:
                     value = 0.05 # Between 0 and 0.1
                if model_orig == models[2] and lang == languages[2] and algo == algorithms[1]:
                     value = 0.1 # Exactly 0.1 (should be shown)

                example_data_list.append({
                    "question_model": model_cleaned, # Use cleaned name in test data
                    "metric_value": value,
                    "language": lang,
                    "retrieval_algorithm_display": algo,
                    "metric_type": metric_to_test
                })

    example_df = pd.DataFrame(example_data_list)

    # In the standalone test, the DataFrame already has cleaned model names.
    # The generator function (which would call clean_model_name) is not used directly here.
    # We need to prepare the test_model_sort_order with cleaned names too.
    test_model_sort_order = [clean_model_name(m) for m in models]
    test_algo_display_order = algorithms

    example_df_metric_specific = example_df[example_df["metric_type"] == metric_to_test].copy()

    # Ensure 'language' column is categorical with the desired order for legend/palette consistency in test
    example_df_metric_specific['language'] = pd.Categorical(
        example_df_metric_specific['language'],
        categories=LANGUAGE_ORDER, # Use the constant order
        ordered=True
    )
    # Ensure 'retrieval_algorithm_display' is categorical for facet order in test
    example_df_metric_specific['retrieval_algorithm_display'] = pd.Categorical(
         example_df_metric_specific['retrieval_algorithm_display'],
         categories=test_algo_display_order,
         ordered=True
    )
    # Ensure 'question_model' is categorical for x-axis order in test
    example_df_metric_specific['question_model'] = pd.Categorical(
         example_df_metric_specific['question_model'],
         categories=test_model_sort_order,
         ordered=True
    )


    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(current_script_dir, "..", "plots_test", "barcharts")
    os.makedirs(test_output_dir, exist_ok=True)

    test_output_path = os.path.join(
        test_output_dir, f"test_model_perf_{metric_to_test}_v5.png" # Incremented version
    )

    print(
        f"\nGenerating example bar chart for: Metric={metric_to_test}"
    )
    create_model_performance_barchart(
        data=example_df_metric_specific,
        output_path=test_output_path,
        metric_name=metric_to_test,
        model_sort_order=test_model_sort_order, # Pass cleaned sort order
        algorithm_display_order=test_algo_display_order,
        figsize_per_facet=(9, 6), # Adjusted for more models and potential long names
        bar_label_fontsize=6, # Using the slightly smaller default
        title_fontsize=16
    )

    print("\n--- Bar Chart Creation Test Finished ---")
    print(f"Test plots (if any) are in: {test_output_dir}")