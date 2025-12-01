import os
import pandas as pd
from typing import List, Optional

from plot_utils import sanitize_filename
from english_barcharts import create_english_retrieval_barchart
from plot_config import clean_model_name

def generate_english_retrieval_comparison_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None
) -> None:
    """
    Generates bar charts comparing Retrieval Algorithms (Hybrid vs Embedding vs Keyword) 
    specifically for English (Markdown files).
    """
    print("\n--- Generating English Retrieval Comparison Bar Charts ---")

    # Filter: English & Markdown
    df_eng = df_data[
        (df_data['language'] == 'english') &
        (df_data['file_extension'] == 'md')
    ].copy()

    if df_eng.empty:
        print("No English/Markdown data found.")
        return

    # Map Algorithms to Display Names
    algo_map = {
        'hybrid': 'Hybrid',
        'embedding': 'Embedding',
        'keyword': 'Keyword'
    }
    df_eng['retrieval_method_display'] = df_eng['retrieval_algorithm'].map(algo_map)
    df_eng = df_eng.dropna(subset=['retrieval_method_display'])

    # Clean Model Names
    df_eng['question_model'] = df_eng['question_model'].apply(clean_model_name)

    # Determine Model Order
    present_models = sorted(df_eng['question_model'].unique())
    final_model_order = present_models
    if model_sort_order:
        cleaned_order = [clean_model_name(m) for m in model_sort_order]
        final_model_order = [m for m in cleaned_order if m in present_models]
        for m in present_models:
            if m not in final_model_order: final_model_order.append(m)

    # Retrieval Method Order
    method_order = ["Hybrid", "Embedding", "Keyword"]
    present_methods = df_eng['retrieval_method_display'].unique()
    final_method_order = [m for m in method_order if m in present_methods]

    metrics = ["f1_score", "accuracy"]

    for metric in metrics:
        df_metric = df_eng[df_eng["metric_type"] == metric]
        if df_metric.empty: continue

        filename = f"{output_filename_prefix}english_retrieval_{sanitize_filename(metric)}.png"
        output_path = os.path.join(output_dir, filename)

        create_english_retrieval_barchart(
            data=df_metric,
            output_path=output_path,
            metric_name=metric,
            model_sort_order=final_model_order,
            retrieval_method_order=final_method_order
        )
