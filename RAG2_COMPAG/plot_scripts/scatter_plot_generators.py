import os
import pandas as pd
from typing import List, Optional

from scatter_plots import create_combined_scatter
from plot_config import clean_model_name, METRIC_DISPLAY_NAMES

def generate_cross_lingual_scatter_plots(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None
):
    """
    Generates combined scatter plots: English Performance (X) vs Average Non-English Performance (Y).
    Combines Markdown, JSON, and XML into one figure.
    """
    print("\n--- Generating Cross-Lingual Scatter Plots (Combined) ---")

    # Filter: Hybrid Algorithm
    df_hybrid = df_data[df_data['retrieval_algorithm'] == 'hybrid'].copy()
    
    if df_hybrid.empty:
        print("No Hybrid data found.")
        return

    # Prepare data structure: Metric -> Format -> DataFrame
    # Actually, we iterate metrics, then collect formats.
    
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}

    # Determine Model Order for consistent coloring
    df_hybrid['question_model'] = df_hybrid['question_model'].apply(clean_model_name)
    
    if model_sort_order:
        cleaned_order = [clean_model_name(m) for m in model_sort_order]
        present_models = df_hybrid['question_model'].unique()
        final_model_order = [m for m in cleaned_order if m in present_models]
        # Append remaining
        for m in sorted(present_models):
            if m not in final_model_order:
                final_model_order.append(m)
    else:
        final_model_order = sorted(df_hybrid['question_model'].unique())

    # Generate for each metric
    for metric in ['f1_score', 'accuracy']:
        data_map = {}
        
        # Collect data for each format
        extensions = sorted(df_hybrid['file_extension'].dropna().unique())
        
        for ext in extensions:
            df_fmt = df_hybrid[df_hybrid['file_extension'] == ext].copy()
            if df_fmt.empty: continue
            
            df_metric = df_fmt[df_fmt['metric_type'] == metric]
            if df_metric.empty: continue

            # Pivot
            pivot = df_metric.pivot_table(
                index='question_model',
                columns='language',
                values='metric_value',
                aggfunc='mean'
            )

            if 'english' not in pivot.columns:
                continue

            non_english_cols = [c for c in pivot.columns if c != 'english']
            if not non_english_cols:
                continue

            plot_data = pd.DataFrame({
                'model': pivot.index,
                'english_score': pivot['english'],
                'non_english_avg': pivot[non_english_cols].mean(axis=1)
            })
            
            # Map extension to display name
            fmt_name = ext_map.get(ext, ext.upper())
            data_map[fmt_name] = plot_data

        if not data_map:
            continue

        # Generate Combined Plot
        metric_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        filename = f"{output_filename_prefix}scatter_cross_lingual_{metric}_combined.png"
        output_path = os.path.join(output_dir, filename)

        create_combined_scatter(
            data_map=data_map,
            output_path=output_path,
            metric_name=metric,
            x_col='english_score',
            y_col='non_english_avg',
            label_col='model',
            title=f"Cross-Lingual Capability ({metric_name})",
            xlabel=f"English {metric_name}",
            ylabel=f"Avg. Non-English {metric_name}",
            hue_order=final_model_order
        )
