import os
import pandas as pd
from typing import List, Optional

from box_plots import create_boxplot
from plot_config import (
    clean_model_name, 
    LANGUAGE_PALETTE, 
    LANGUAGE_ORDER,
    METRIC_DISPLAY_NAMES
)

def generate_f1_distribution_boxplot(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None
):
    """
    Generates F1 Score distribution boxplots.
    X-Axis: Model
    Hue: Language
    Filter: Hybrid RAG & Markdown.
    """
    print("\n--- Generating F1 Distribution Boxplots (Hybrid/Markdown) ---")

    # Filter: Hybrid & Markdown
    df_plot = df_data[
        (df_data['retrieval_algorithm'] == 'hybrid') & 
        (df_data['file_extension'] == 'md') &
        (df_data['metric_type'] == 'f1_score')
    ].copy()

    if df_plot.empty:
        print("No Hybrid/Markdown F1 data found.")
        return

    # Clean Names
    df_plot['question_model'] = df_plot['question_model'].apply(clean_model_name)

    # Determine Order
    present_models = sorted(df_plot['question_model'].unique())
    final_order = present_models
    if model_sort_order:
        cleaned_order = [clean_model_name(m) for m in model_sort_order]
        final_order = [m for m in cleaned_order if m in present_models]
        # Append remaining
        for m in present_models:
            if m not in final_order: final_order.append(m)

    output_path = os.path.join(output_dir, f"{output_filename_prefix}boxplot_f1_distribution.png")

    create_boxplot(
        data=df_plot,
        output_path=output_path,
        x_col="question_model",
        y_col="metric_value",
        hue_col="language", # Color by language to show spread across languages
        metric_name="f1_score",
        title="F1 Score Distribution per Model (Across Languages)",
        order=final_order,
        hue_order=LANGUAGE_ORDER,
        palette=LANGUAGE_PALETTE
    )
