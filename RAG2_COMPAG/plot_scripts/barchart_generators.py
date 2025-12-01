import os
import pandas as pd
from typing import List, Optional

from plot_utils import sanitize_filename
from barcharts import create_grouped_barchart
from plot_config import (
    LANGUAGE_PALETTE, LANGUAGE_ORDER, 
    FORMAT_PALETTE, FORMAT_ORDER,
    clean_model_name, METRIC_DISPLAY_NAMES
)

def generate_model_performance_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None
) -> None:
    """
    Generates bar charts for Hybrid RAG performance: Model vs Language.
    Iterates over all available file formats (md, xml, json).
    """
    print("\n--- Generating Model Performance Bar Charts (Hybrid / per Format) ---")

    # Filter: Hybrid Algorithm
    df_hybrid = df_data[df_data["retrieval_algorithm"] == "hybrid"].copy()
    
    if df_hybrid.empty:
        print("No Hybrid data found.")
        return

    # Identify available extensions
    extensions = sorted(df_hybrid["file_extension"].dropna().unique())
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}

    for ext in extensions:
        print(f"Processing format: {ext}")
        
        df_plot = df_hybrid[df_hybrid["file_extension"] == ext].copy()
        if df_plot.empty: continue
    
        df_plot['question_model'] = df_plot['question_model'].apply(clean_model_name)
        
        # Determine Model Order
        if model_sort_order:
            cleaned_order = [clean_model_name(m) for m in model_sort_order]
            present_models = df_plot['question_model'].unique()
            final_model_order = [m for m in cleaned_order if m in present_models]
            for m in sorted(present_models):
                if m not in final_model_order:
                    final_model_order.append(m)
        else:
            final_model_order = sorted(df_plot['question_model'].unique())

        ext_display = ext_map.get(ext, ext.upper())

        # Generate plots for each metric
        for metric in ["f1_score", "accuracy"]:
            df_metric = df_plot[df_plot["metric_type"] == metric]
            if df_metric.empty: 
                continue

            # Include extension in filename
            filename = f"{output_filename_prefix}hybrid_lang_perf_{sanitize_filename(metric)}_{ext}.png"
            output_path = os.path.join(output_dir, filename)

            create_grouped_barchart(
                data=df_metric,
                output_path=output_path,
                metric_name=metric,
                x_col="question_model",
                y_col="metric_value",
                hue_col="language",
                x_order=final_model_order,
                hue_order=LANGUAGE_ORDER,
                palette=LANGUAGE_PALETTE,
                title=f"Hybrid RAG Performance by Language ({METRIC_DISPLAY_NAMES.get(metric, metric)}) - {ext_display}",
                xlabel="LLM Model"
            )


def generate_format_comparison_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None
) -> None:
    """
    Generates bar charts for Hybrid RAG performance: Model vs File Format.
    Uses Hybrid algorithm. 
    1. Aggregates over all languages.
    2. Generates specific plot for English.
    """
    print("\n--- Generating Format Comparison Bar Charts (Hybrid) ---")

    # Filter: Hybrid Algorithm only
    df_plot = df_data[df_data["retrieval_algorithm"] == "hybrid"].copy()
    
    if df_plot.empty:
        print("No Hybrid data found for format comparison.")
        return

    # Map file extensions to display names
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}
    df_plot['format_display'] = df_plot['file_extension'].map(ext_map)
    # Remove rows where format mapping failed
    df_plot = df_plot.dropna(subset=['format_display'])
    
    df_plot['question_model'] = df_plot['question_model'].apply(clean_model_name)

    # Determine Model Order (reuse logic)
    present_models = df_plot['question_model'].unique()
    if model_sort_order:
        cleaned_order = [clean_model_name(m) for m in model_sort_order]
        final_model_order = [m for m in cleaned_order if m in present_models]
        for m in sorted(present_models):
            if m not in final_model_order:
                final_model_order.append(m)
    else:
        final_model_order = sorted(present_models)

    for metric in ["f1_score", "accuracy"]:
        df_metric = df_plot[df_plot["metric_type"] == metric]
        if df_metric.empty:
            continue
        
        metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)

        # --- 1. Average across all languages ---
        filename_avg = f"{output_filename_prefix}hybrid_format_perf_{sanitize_filename(metric)}_avg_all_langs.png"
        output_path_avg = os.path.join(output_dir, filename_avg)

        create_grouped_barchart(
            data=df_metric,
            output_path=output_path_avg,
            metric_name=metric,
            x_col="question_model",
            y_col="metric_value",
            hue_col="format_display",
            x_order=final_model_order,
            hue_order=FORMAT_ORDER,
            palette=FORMAT_PALETTE,
            title=f"Hybrid RAG Performance by Data Format ({metric_display}) - Avg across all languages",
            xlabel="LLM Model"
        )

        # --- 2. English Only ---
        df_english = df_metric[df_metric["language"] == "english"]
        
        if not df_english.empty:
            filename_eng = f"{output_filename_prefix}hybrid_format_perf_{sanitize_filename(metric)}_english.png"
            output_path_eng = os.path.join(output_dir, filename_eng)

            create_grouped_barchart(
                data=df_english,
                output_path=output_path_eng,
                metric_name=metric,
                x_col="question_model",
                y_col="metric_value",
                hue_col="format_display",
                x_order=final_model_order,
                hue_order=FORMAT_ORDER,
                palette=FORMAT_PALETTE,
                title=f"Hybrid RAG Performance by Data Format ({metric_display}) - English",
                xlabel="LLM Model"
            )
