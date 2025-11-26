import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import numpy as np 
# [Import clean_model_name for standalone test consistency]
try:
    from .plot_utils import clean_model_name
except ImportError:
     # Fallback if running directly
     def clean_model_name(x): return x

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
# Updated to include Dutch, Spanish, Italian
LANGUAGE_ORDER = ["english", "french", "german", "dutch", "spanish", "italian"]
LANGUAGE_PALETTE = {
    "english": "#1f77b4",  # Muted blue
    "french": "#ff7f0e",   # Safety orange
    "german": "#2ca02c",   # Cooked asparagus green
    "dutch": "#9467bd",    # Muted purple
    "spanish": "#d62728",  # Brick red
    "italian": "#17becf"   # Cyan/Teal
}
# A fallback color for any unexpected languages
DEFAULT_LANGUAGE_COLOR = "#7f7f7f" # Medium gray

# Mapping for language full name to short code for labels
LANGUAGE_CODES = {
    "english": "EN",
    "french": "FR",
    "german": "DE",
    "dutch": "NL",
    "spanish": "ES",
    "italian": "IT"
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
    figsize_per_facet: Tuple[float, float] = (7.5, 5.5), 
    bar_label_fontsize: int = 6, 
    title_fontsize: int = 16,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int = 10
) -> None:
    """
    Creates a grouped bar chart showing a specific metric for different models,
    grouped by language, and faceted by retrieval algorithm.
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
    actual_languages_in_data = sorted(plot_data['language'].unique())
    
    current_lang_order = [lang for lang in language_order if lang in actual_languages_in_data]
    for lang in actual_languages_in_data: 
        if lang not in current_lang_order:
            current_lang_order.append(lang)
        if lang not in current_palette:
            current_palette[lang] = DEFAULT_LANGUAGE_COLOR
            if lang not in language_order: 
                language_order = language_order + [lang]


    # Determine the number of facets for figure size calculation
    num_facets = plot_data['retrieval_algorithm_display'].nunique()
    if num_facets == 0:
        print(f"Warning: No retrieval algorithms found in data for {metric_display_name}. Skipping plot.")
        return
    
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
            hue_order=current_lang_order, 
            palette=current_palette,      
            col_order=algorithm_display_order,
            height=fig_height,
            aspect=figsize_per_facet[0] / fig_height, 
            legend=False, 
            sharex=True, 
            sharey=True
        )

        num_facet_cols = g.axes.shape[1] if g.axes.ndim == 2 else len(g.axes)

        first_ax = g.axes.flat[0]
        x_tick_locations = first_ax.get_xticks() if first_ax.get_xticks() is not None else []
        x_tick_labels = [label.get_text() for label in first_ax.get_xticklabels()] if first_ax.get_xticklabels() is not None else []

        draw_labels = True
        if len(x_tick_locations) != len(x_tick_labels) or not x_tick_locations:
             print(f"Warning: Cannot reliably position labels due to issues with x-ticks. Skipping label drawing.")
             draw_labels = False


        for ax_idx, ax in enumerate(g.axes.flat):
            facet_algorithm_display = ax.get_title().split('=')[-1].strip()
            
            y_min_ax, y_max_ax = ax.get_ylim()
            label_above_offset = (y_max_ax - y_min_ax) * 0.01 


            if draw_labels:
                model_patches_map = {}
                for patch in ax.patches:
                     patch_center_x = patch.get_x() + patch.get_width()/2.
                     try:
                         valid_x_tick_locations = [loc for i, loc in enumerate(x_tick_locations) if i < len(x_tick_labels)]
                         if not valid_x_tick_locations: continue 

                         closest_model_idx = np.argmin(np.abs(valid_x_tick_locations - patch_center_x))
                         model_name = x_tick_labels[closest_model_idx]
                     except ValueError:
                         continue 

                     if model_name not in model_patches_map:
                          model_patches_map[model_name] = []
                     model_patches_map[model_name].append(patch)

                for model_name in x_tick_labels: 
                     if model_name in model_patches_map:
                          sorted_patches = sorted(model_patches_map[model_name], key=lambda p: p.get_x())

                          for hue_index, patch in enumerate(sorted_patches):
                              height = patch.get_height()

                              if pd.notna(height) and height >= 0:
                                   center_x = patch.get_x() + patch.get_width() / 2.
                                   metric_text_y = max(0, height) + label_above_offset 
                                   ax.text(
                                       center_x,
                                       metric_text_y,
                                       f"{height:.2f}", 
                                       ha="center",
                                       va="bottom", 
                                       fontsize=bar_label_fontsize,
                                       color="black", 
                                       rotation=0 
                                   )

                                   try:
                                       lang = current_lang_order[hue_index]
                                       lang_code = LANGUAGE_CODES.get(lang, lang.upper()[:2]) 
                                   except IndexError:
                                       lang_code = DEFAULT_LANGUAGE_CODE 
                                       lang = "Unknown" 

                                   lang_text_y = height * 0.5 
                                   lang_label_color = "white"
                                   lang_label_va = "center"
                                   
                                   # Only draw text inside if bar is tall enough to possibly contain it
                                   if height > 0.1:
                                       ax.text(
                                           center_x,
                                           lang_text_y,
                                           lang_code, 
                                           ha="center",
                                           va=lang_label_va, 
                                           fontsize=bar_label_fontsize,
                                           color=lang_label_color, 
                                           weight='bold' 
                                       )

            # Set y-axis limits
            facet_plot_data = plot_data[plot_data['retrieval_algorithm_display'] == facet_algorithm_display]
            max_height_in_facet = facet_plot_data['metric_value'].max()

            if pd.notna(max_height_in_facet):
                 buffer_space = label_above_offset + 0.02 
                 upper_ylim = max(1.05, max_height_in_facet + buffer_space)
            else:
                 upper_ylim = 1.05 

            ax.set_ylim(0, upper_ylim)
            ax.set_xlabel("Question Model", fontsize=axis_label_fontsize)
            
            if ax_idx % num_facet_cols == 0:
                 ax.set_ylabel(metric_display_name, fontsize=axis_label_fontsize)
            else:
                 ax.set_ylabel("") 

            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.tick_params(axis='y', labelsize=tick_label_fontsize)

        plt.tight_layout(rect=[0, 0.03, 0.9, 0.97]) 

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
    # Test block updated to reflect new structure if needed, or kept simple
    print("--- Testing Model Performance Bar Chart Creation (Standalone) ---")
    pass