import os
import pandas as pd
from plot_scripts.heatmaps import create_heatmap
from plot_scripts.plot_utils import sanitize_filename

def generate_global_overview_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Generates Global Overview heatmaps: 
    X-Axis: Languages
    Y-Axis: Models
    Metrics: Accuracy and F1 Score
    
    NOTE: Filters for Markdown ('md') files to ensure consistent comparison across languages.
    """
    print("\n--- Generating Global Overview Heatmaps (Lang vs Model) ---")
    
    # Filter for Markdown only to avoid format bias in global average
    if 'file_extension' in df.columns:
        df_filtered = df[df['file_extension'] == 'md'].copy()
        if df_filtered.empty:
             print("Warning: No Markdown files found. Using all data for global heatmap.")
             df_filtered = df.copy()
        else:
             print("Filtered data to 'md' extension for consistent global comparison.")
    else:
        df_filtered = df.copy()

    metrics = {
        'f1_score': 'F1 Score',
        'accuracy': 'Accuracy'
    }

    for metric_key, metric_label in metrics.items():
        # Filter data for the specific metric
        df_metric = df_filtered[df_filtered['metric_type'] == metric_key].copy()
        
        if df_metric.empty:
            print(f"No data found for {metric_label}. Skipping.")
            continue

        output_filename = f"global_heatmap_lang_vs_model_{metric_key}.png"
        output_path = os.path.join(output_dir, output_filename)

        create_heatmap(
            data=df_metric,
            output_path=output_path,
            index_col='question_model',   # Y-Axis
            columns_col='language',       # X-Axis
            values_col='metric_value',
            metric_label=metric_label,
            title=f"Global Overview: {metric_label} (Model vs Language - Markdown)",
            xlabel="Language",
            ylabel="LLM Model"
        )

def generate_format_comparison_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Generates Format Comparison heatmaps (Format vs Model).
    Aggregates across languages/algorithms to show general format performance.
    """
    print("\n--- Generating Format Comparison Heatmaps (Format vs Model) ---")
    
    if 'file_extension' not in df.columns:
        print("Column 'file_extension' missing. Skipping format heatmaps.")
        return

    metrics = {
        'f1_score': 'F1 Score',
        'accuracy': 'Accuracy'
    }

    for metric_key, metric_label in metrics.items():
        df_metric = df[df['metric_type'] == metric_key].copy()
        
        if df_metric.empty:
            continue

        output_filename = f"global_heatmap_format_vs_model_{metric_key}.png"
        output_path = os.path.join(output_dir, output_filename)

        create_heatmap(
            data=df_metric,
            output_path=output_path,
            index_col='question_model',   # Y-Axis
            columns_col='file_extension', # X-Axis
            values_col='metric_value',
            metric_label=metric_label,
            title=f"Format Analysis: {metric_label} (Model vs File Format)",
            xlabel="File Format",
            ylabel="LLM Model"
        )

def generate_english_format_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Generates English-only heatmaps:
    X-Axis: File Formats (md, json, xml)
    Y-Axis: Models
    Metrics: Accuracy and F1 Score
    """
    print("\n--- Generating English Format Heatmaps (Format vs Model) ---")
    
    # Filter for English only
    df_english = df[df['language'] == 'english'].copy()
    
    if df_english.empty:
        print("No English data found. Skipping English Format heatmaps.")
        return

    metrics = {
        'f1_score': 'F1 Score',
        'accuracy': 'Accuracy'
    }

    for metric_key, metric_label in metrics.items():
        df_metric = df_english[df_english['metric_type'] == metric_key].copy()
        
        if df_metric.empty:
            print(f"No English data found for {metric_label}. Skipping.")
            continue

        output_filename = f"english_heatmap_format_vs_model_{metric_key}.png"
        output_path = os.path.join(output_dir, output_filename)

        create_heatmap(
            data=df_metric,
            output_path=output_path,
            index_col='question_model',   # Y-Axis
            columns_col='file_extension', # X-Axis (Format)
            values_col='metric_value',
            metric_label=metric_label,
            title=f"English Analysis: {metric_label} (Model vs File Format)",
            xlabel="File Format",
            ylabel="LLM Model"
        )

def generate_markdown_overview_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Generates Markdown-only Overview heatmaps (Redundant with global overview if filtered, 
    but kept for explicit requests).
    """
    print("\n--- Generating Markdown Overview Heatmaps (Lang vs Model) ---")
    
    # Filter for Markdown only
    # Assumes file_extension is stored as 'md' (without dot)
    if 'file_extension' in df.columns:
        df_md = df[df['file_extension'] == 'md'].copy()
    else:
        df_md = pd.DataFrame() # Empty
    
    if df_md.empty:
        print("No Markdown data found. Skipping Markdown Overview heatmaps.")
        return

    metrics = {
        'f1_score': 'F1 Score',
        'accuracy': 'Accuracy'
    }

    for metric_key, metric_label in metrics.items():
        df_metric = df_md[df_md['metric_type'] == metric_key].copy()
        
        if df_metric.empty:
            continue

        output_filename = f"markdown_heatmap_lang_vs_model_{metric_key}.png"
        output_path = os.path.join(output_dir, output_filename)

        create_heatmap(
            data=df_metric,
            output_path=output_path,
            index_col='question_model',   # Y-Axis
            columns_col='language',       # X-Axis
            values_col='metric_value',
            metric_label=metric_label,
            title=f"Markdown Overview: {metric_label} (Model vs Language)",
            xlabel="Language",
            ylabel="LLM Model"
        )
