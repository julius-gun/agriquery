import os
import pandas as pd
from typing import List, Optional

from scatter_plots import create_combined_scatter, create_efficiency_scatter, create_gap_line_plot
from plot_config import clean_model_name, METRIC_DISPLAY_NAMES, MODEL_PARAMS, LANGUAGE_PALETTE

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

def generate_model_efficiency_plot(
    df_data: pd.DataFrame,
    output_dir: str
):
    """
    Generates scatter plots: Model Size (Parameters) vs Performance.
    Uses 'hybrid' algorithm, Markdown format, and results for English only.
    Generates for: F1 Score and Precision.
    """
    print("\n--- Generating Model Efficiency Plots (Size vs Performance) ---")
    
    metrics_to_plot = ['f1_score', 'precision']

    for metric in metrics_to_plot:
        # Filter: Hybrid, Markdown (preferred format), Metric, English Only
        df = df_data[
            (df_data['retrieval_algorithm'] == 'hybrid') & 
            (df_data['file_extension'] == 'md') &
            (df_data['metric_type'] == metric) &
            (df_data['language'] == 'english')
        ].copy()
        
        if df.empty:
            print(f"No English Markdown data for Efficiency Plot ({metric}).")
            continue

        # Clean Model Names
        df['question_model'] = df['question_model'].apply(clean_model_name)
        
        # Group by model and take mean score (in case of duplicates)
        df_agg = df.groupby('question_model')['metric_value'].mean().reset_index()
        
        # Map Parameters
        df_agg['parameters'] = df_agg['question_model'].map(MODEL_PARAMS)
        
        # Drop models without known parameters
        missing = df_agg[df_agg['parameters'].isna()]['question_model'].tolist()
        if missing:
            print(f"Warning: Missing parameter counts for models: {missing}. Excluded from {metric} plot.")
        
        df_agg = df_agg.dropna(subset=['parameters'])
        
        if df_agg.empty:
            print(f"No valid models for Efficiency Plot ({metric}).")
            continue
            
        metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)
        filename = f"scatter_size_vs_{metric}.png"
        output_path = os.path.join(output_dir, filename)
        
        create_efficiency_scatter(
            df=df_agg,
            output_path=output_path,
            x_col='parameters',
            y_col='metric_value',
            label_col='question_model',
            title=f"Model Efficiency: Parameter Count vs. {metric_display} (English Markdown)",
            ylabel=metric_display
        )

def generate_performance_gap_plot(
    df_data: pd.DataFrame,
    output_dir: str
):
    """
    Generates a performance comparison plot across languages for Markdown.
    Line chart: Models (X-axis, sorted by size) vs F1 Score (Y-axis) for all languages.
    """
    print("\n--- Generating Performance Gap Plot (All Languages) ---")
    
    # Use valid languages from palette to ensure consistent coloring
    valid_langs = list(LANGUAGE_PALETTE.keys())
    
    # Filter: Hybrid, Markdown, F1 Score, Valid Languages
    df = df_data[
        (df_data['retrieval_algorithm'] == 'hybrid') & 
        (df_data['file_extension'] == 'md') &
        (df_data['metric_type'] == 'f1_score') &
        (df_data['language'].isin(valid_langs))
    ].copy()
    
    if df.empty:
        print("No data for Gap Plot.")
        return

    df['question_model'] = df['question_model'].apply(clean_model_name)
    
    # Aggregate (in case multiple runs)
    df_agg = df.groupby(['question_model', 'language'])['metric_value'].mean().reset_index()
    
    # Determine Sort Order: By Parameters
    # We need a list of models sorted by params
    unique_models = df_agg['question_model'].unique()
    
    # Sort models by params (using a large default for unknown to push to end)
    model_order = sorted(unique_models, key=lambda m: MODEL_PARAMS.get(m, 9999))
    
    # Convert model column to categorical with this order for plotting
    df_agg['question_model'] = pd.Categorical(df_agg['question_model'], categories=model_order, ordered=True)
    df_agg = df_agg.sort_values('question_model')
    
    output_path = os.path.join(output_dir, "line_performance_gap_markdown.png")
    
    create_gap_line_plot(
        df=df_agg,
        output_path=output_path,
        x_col='question_model',
        y_col='metric_value',
        hue_col='language',
        title="Cross-Lingual Performance: F1 Score by Model (Markdown)",
        model_order=model_order
    )
