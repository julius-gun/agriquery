import os
import pandas as pd
from typing import List, Optional, Tuple

from plot_utils import sanitize_filename
from heatmaps import create_heatmap
from plot_config import (
    clean_model_name, 
    LANGUAGE_ORDER, 
    FORMAT_ORDER,
    METRIC_DISPLAY_NAMES
)

def _prepare_data(df: pd.DataFrame, metric: str, model_sort_order: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """Helper to clean model names and determine order."""
    df_clean = df.copy()
    df_clean['question_model'] = df_clean['question_model'].apply(clean_model_name)
    
    present_models = df_clean['question_model'].unique()
    final_order = sorted(present_models)
    
    if model_sort_order:
        cleaned_order = [clean_model_name(m) for m in model_sort_order]
        final_order = [m for m in cleaned_order if m in present_models]
        for m in sorted(present_models):
            if m not in final_order:
                final_order.append(m)
                
    return df_clean, final_order

def generate_global_overview_heatmaps(df: pd.DataFrame, output_dir: str, model_sort_order: Optional[List[str]] = None):
    """
    Language vs Model.
    Iterates over all available file formats (md, xml, json).
    """
    print("\n--- Generating Global Overview Heatmaps (Hybrid / per Format) ---")
    
    # Filter: Hybrid only
    df_hybrid = df[df['retrieval_algorithm'] == 'hybrid'].copy()
    
    if df_hybrid.empty:
        print("No Hybrid data found.")
        return

    extensions = sorted(df_hybrid["file_extension"].dropna().unique())
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}

    for ext in extensions:
        df_filtered = df_hybrid[df_hybrid['file_extension'] == ext].copy()
        if df_filtered.empty: continue

        ext_display = ext_map.get(ext, ext.upper())

        for metric in ["f1_score", "accuracy"]:
            df_metric = df_filtered[df_filtered['metric_type'] == metric]
            if df_metric.empty: continue

            df_clean, final_model_order = _prepare_data(df_metric, metric, model_sort_order)
            
            # Filename with extension
            output_path = os.path.join(output_dir, f"heatmap_language_vs_model_{metric}_{ext}.png")

            create_heatmap(
                data=df_clean,
                output_path=output_path,
                index_col='question_model',
                columns_col='language',
                values_col='metric_value',
                metric_name=metric,
                title=f"Hybrid RAG Performance: Model vs Language ({METRIC_DISPLAY_NAMES.get(metric)}) - {ext_display}",
                xlabel="Language",
                ylabel="LLM Model",
                index_order=final_model_order,
                columns_order=LANGUAGE_ORDER
            )

def generate_format_comparison_heatmaps(df: pd.DataFrame, output_dir: str, model_sort_order: Optional[List[str]] = None):
    """
    Format vs Model.
    Filter: Hybrid Algorithm (Aggregated across languages).
    """
    print("\n--- Generating Format Comparison Heatmaps (Hybrid) ---")
    
    df_filtered = df[df['retrieval_algorithm'] == 'hybrid'].copy()
    
    if df_filtered.empty:
        print("No Hybrid data found.")
        return

    # Map extensions
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}
    df_filtered['format_display'] = df_filtered['file_extension'].map(ext_map)
    df_filtered = df_filtered.dropna(subset=['format_display'])

    for metric in ["f1_score", "accuracy"]:
        df_metric = df_filtered[df_filtered['metric_type'] == metric]
        if df_metric.empty: continue

        df_clean, final_model_order = _prepare_data(df_metric, metric, model_sort_order)

        output_path = os.path.join(output_dir, f"heatmap_format_vs_model_{metric}.png")

        create_heatmap(
            data=df_clean,
            output_path=output_path,
            index_col='question_model',
            columns_col='format_display',
            values_col='metric_value',
            metric_name=metric,
            title=f"Hybrid RAG Performance: Model vs Format ({METRIC_DISPLAY_NAMES.get(metric)})",
            xlabel="File Format",
            ylabel="LLM Model",
            index_order=final_model_order,
            columns_order=FORMAT_ORDER
        )
