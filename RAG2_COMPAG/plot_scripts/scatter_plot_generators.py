import os
import pandas as pd
from typing import List, Optional

from scatter_plots import create_comparison_scatter
from plot_config import clean_model_name, METRIC_DISPLAY_NAMES

def generate_cross_lingual_scatter_plots(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None
):
    """
    Generates scatter plots: English Performance (X) vs Average Non-English Performance (Y).
    Iterates over file formats (MD, XML, JSON).
    """
    print("\n--- Generating Cross-Lingual Scatter Plots ---")

    # Filter: Hybrid Algorithm
    df_hybrid = df_data[df_data['retrieval_algorithm'] == 'hybrid'].copy()
    
    if df_hybrid.empty:
        print("No Hybrid data found.")
        return

    # Identify formats
    extensions = sorted(df_hybrid['file_extension'].dropna().unique())
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}

    for ext in extensions:
        df_fmt = df_hybrid[df_hybrid['file_extension'] == ext].copy()
        if df_fmt.empty: continue
        
        ext_display = ext_map.get(ext, ext.upper())
        
        # Clean Model Names
        df_fmt['question_model'] = df_fmt['question_model'].apply(clean_model_name)

        # We focus on F1 Score and Accuracy
        for metric in ['f1_score', 'accuracy']:
            df_metric = df_fmt[df_fmt['metric_type'] == metric]
            if df_metric.empty: continue

            # Pivot to get Language columns
            # Index: Model, Columns: Language, Values: Metric
            pivot = df_metric.pivot_table(
                index='question_model',
                columns='language',
                values='metric_value',
                aggfunc='mean'
            )

            if 'english' not in pivot.columns:
                continue

            # Calculate English vs Non-English Avg
            non_english_cols = [c for c in pivot.columns if c != 'english']
            if not non_english_cols:
                continue

            plot_data = pd.DataFrame({
                'model': pivot.index,
                'english_score': pivot['english'],
                'non_english_avg': pivot[non_english_cols].mean(axis=1)
            })

            filename = f"{output_filename_prefix}scatter_cross_lingual_{metric}_{ext}.png"
            output_path = os.path.join(output_dir, filename)
            
            metric_name = METRIC_DISPLAY_NAMES.get(metric, metric)

            create_comparison_scatter(
                data=plot_data,
                output_path=output_path,
                x_col='english_score',
                y_col='non_english_avg',
                label_col='model',
                metric_name=metric,
                title=f"Cross-Lingual Capability ({metric_name}) - {ext_display}",
                xlabel=f"English {metric_name}",
                ylabel=f"Avg. Non-English {metric_name}"
            )