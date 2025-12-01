import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Tuple

# Assuming plot_utils.py and barcharts.py are in the same directory or accessible
try:
    from .plot_utils import add_project_paths, sanitize_filename, get_model_colors
    from .barcharts import create_model_performance_barchart, LANGUAGE_ORDER, LANGUAGE_PALETTE
except ImportError:
    # Fallback for standalone execution or when run from a different directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import add_project_paths, sanitize_filename, get_model_colors
    from barcharts import create_model_performance_barchart, LANGUAGE_ORDER, LANGUAGE_PALETTE

add_project_paths()

from visualization_data_extractor import extract_detailed_visualization_data
from utils.config_loader import ConfigLoader

# Metrics to generate bar charts for
TARGET_METRICS_FOR_BARCHART = ["f1_score", "accuracy"]

# RAG algorithms to include in the comparison.
RAG_ALGORITHMS_TO_PLOT = {
    "hybrid": "Hybrid RAG",
    "embedding": "Embedding RAG",
    "keyword": "Keyword RAG",
}

# Configuration for the "Full Manual" data point
FULL_MANUAL_ALIAS = "Full Manual"
FULL_MANUAL_NOISE_LEVEL = 59000 # zeroshot at this noise level

# Order in which algorithm facets should appear
DEFAULT_ALGORITHM_DISPLAY_ORDER = ["Hybrid", "Embedding", "Keyword", FULL_MANUAL_ALIAS]

# Specific model name mappings for beautification
MODEL_NAME_MAPPINGS = {
    "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
    "phi3_14B_q4_medium-128k": "phi3 14B",
}

def clean_model_name(model_name: str) -> str:
    """
    Applies specific cleaning rules to a model name:
    """
    cleaned_name = MODEL_NAME_MAPPINGS.get(model_name, model_name)

    if cleaned_name == model_name:
        if cleaned_name.endswith("-128k"):
            cleaned_name = cleaned_name.removesuffix("-128k")
        cleaned_name = cleaned_name.replace("_", " ")

    return cleaned_name

def generate_model_performance_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None,
    figsize_per_facet: Tuple[float, float] = (7, 5.5)
) -> None:
    """
    Generates grouped bar charts comparing model performance (various metrics)
    for selected RAG algorithms and "Full Manual".
    
    IMPORTANT: Filters for Markdown ('md') files only to ensure apples-to-apples comparison.
    """
    print("\n--- Generating Model Performance Bar Charts (Markdown Only) ---")

    required_cols = [
        "language", "question_model", "retrieval_algorithm", "noise_level",
        "metric_type", "metric_value"
    ]
    if not all(col in df_data.columns for col in required_cols):
        print(f"Error: Input DataFrame is missing one or more required columns. Found: {df_data.columns.tolist()}")
        return

    df_plot_base = df_data.copy()
    
    # Filter for Markdown files if extension info is available
    if 'file_extension' in df_plot_base.columns:
        original_count = len(df_plot_base)
        df_plot_base = df_plot_base[df_plot_base['file_extension'] == 'md'].copy()
        print(f"Filtered data to 'md' extension: {len(df_plot_base)}/{original_count} rows.")
    
    df_plot_base['question_model'] = df_plot_base['question_model'].apply(clean_model_name)

    # --- 1. Prepare "Full Manual" data ---
    df_full_manual = df_plot_base[
        (df_plot_base["retrieval_algorithm"] == "zeroshot") &
        (df_plot_base["noise_level"] == FULL_MANUAL_NOISE_LEVEL)
    ].copy()

    if not df_full_manual.empty:
        df_full_manual["retrieval_algorithm_display"] = FULL_MANUAL_ALIAS

    # --- 2. Prepare RAG data ---
    rag_data_frames = []
    present_rag_algorithms_for_plot = []
    for algo_internal_name, algo_display_name in RAG_ALGORITHMS_TO_PLOT.items():
        df_rag_algo = df_plot_base[df_plot_base["retrieval_algorithm"] == algo_internal_name].copy()
        if not df_rag_algo.empty:
            df_rag_algo["retrieval_algorithm_display"] = algo_display_name
            rag_data_frames.append(df_rag_algo)
            present_rag_algorithms_for_plot.append(algo_display_name)
    
    # --- 3. Determine Facet Order ---
    all_present_algorithms_set = set(present_rag_algorithms_for_plot)
    if not df_full_manual.empty:
        all_present_algorithms_set.add(FULL_MANUAL_ALIAS)

    if not all_present_algorithms_set:
        print("No RAG data or 'Full Manual' data found after filtering. Skipping bar chart generation.")
        return

    final_algorithm_display_order = []
    for algo_display_name in DEFAULT_ALGORITHM_DISPLAY_ORDER:
        if algo_display_name in all_present_algorithms_set:
            final_algorithm_display_order.append(algo_display_name)
            all_present_algorithms_set.remove(algo_display_name)

    for algo_display_name in sorted(list(all_present_algorithms_set)):
        final_algorithm_display_order.append(algo_display_name)
        
    # --- 4. Combine data ---
    all_plot_data_frames = rag_data_frames
    if not df_full_manual.empty:
        all_plot_data_frames.append(df_full_manual)

    if not all_plot_data_frames:
        return

    df_combined = pd.concat(all_plot_data_frames, ignore_index=True)

    # --- 5. Clean Model Sort Order ---
    cleaned_model_sort_order = None
    if model_sort_order is not None:
         cleaned_model_sort_order = [clean_model_name(model) for model in model_sort_order]
         models_in_cleaned_data = df_combined['question_model'].unique().tolist()
         cleaned_model_sort_order = [model for model in cleaned_model_sort_order if model in models_in_cleaned_data]

    # --- 6. Generate plots ---
    for metric_name in TARGET_METRICS_FOR_BARCHART:
        df_metric_specific = df_combined[df_combined["metric_type"] == metric_name].copy()

        if df_metric_specific.empty:
            continue
        
        # Determine language order dynamically
        actual_languages = sorted(df_metric_specific['language'].unique())
        current_lang_order = [l for l in LANGUAGE_ORDER if l in actual_languages]
        for l in actual_languages:
            if l not in current_lang_order: current_lang_order.append(l)

        plot_data_for_metric = df_metric_specific[[
            "question_model", "metric_value", "language", "retrieval_algorithm_display"
        ]].copy()

        sanitized_metric = sanitize_filename(metric_name)
        filename = f"{output_filename_prefix}model_perf_{sanitized_metric}_comparison.png"
        output_filepath = os.path.join(output_dir, filename)

        create_model_performance_barchart(
            data=plot_data_for_metric,
            output_path=output_filepath,
            metric_name=metric_name,
            model_sort_order=cleaned_model_sort_order,
            algorithm_display_order=final_algorithm_display_order,
            language_order=current_lang_order,
            language_palette=LANGUAGE_PALETTE,
            figsize_per_facet=figsize_per_facet
        )

def generate_format_comparison_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None,
    figsize_per_facet: Tuple[float, float] = (7, 5.5)
) -> None:
    """
    Generates grouped bar charts comparing FILE FORMATS (Markdown, XML, JSON).
    Focuses on 'Hybrid' RAG algorithm to analyze format impact.
    """
    print("\n--- Generating Format Comparison Bar Charts ---")

    if 'file_extension' not in df_data.columns:
        print("Error: 'file_extension' column missing. Cannot generate format comparison.")
        return

    df_plot = df_data.copy()
    df_plot['question_model'] = df_plot['question_model'].apply(clean_model_name)

    # Filter for Hybrid RAG only (primary focus for format comparison)
    target_algo = "hybrid"
    df_hybrid = df_plot[df_plot["retrieval_algorithm"] == target_algo].copy()
    
    if df_hybrid.empty:
        print(f"No data found for algorithm '{target_algo}'. Skipping format comparison.")
        return

    # Map file extensions to display names
    ext_mapping = {'md': 'Markdown', 'xml': 'XML', 'json': 'JSON'}
    df_hybrid['retrieval_algorithm_display'] = df_hybrid['file_extension'].map(ext_mapping).fillna(df_hybrid['file_extension'])

    # Determine Display Order
    display_order = ['Markdown', 'XML', 'JSON']
    available_formats = df_hybrid['retrieval_algorithm_display'].unique()
    final_display_order = [fmt for fmt in display_order if fmt in available_formats]

    # Clean Sort Order
    cleaned_model_sort_order = None
    if model_sort_order:
         cleaned_model_sort_order = [clean_model_name(model) for model in model_sort_order]
         models_in_data = df_hybrid['question_model'].unique().tolist()
         cleaned_model_sort_order = [model for model in cleaned_model_sort_order if model in models_in_data]

    for metric_name in TARGET_METRICS_FOR_BARCHART:
        df_metric = df_hybrid[df_hybrid["metric_type"] == metric_name].copy()
        if df_metric.empty:
            continue
        
        # Determine language order
        actual_languages = sorted(df_metric['language'].unique())
        current_lang_order = [l for l in LANGUAGE_ORDER if l in actual_languages]
        for l in actual_languages:
            if l not in current_lang_order: current_lang_order.append(l)

        plot_data = df_metric[[
            "question_model", "metric_value", "language", "retrieval_algorithm_display"
        ]].copy()

        sanitized_metric = sanitize_filename(metric_name)
        # Unique filename for format comparison
        filename = f"{output_filename_prefix}format_perf_{sanitized_metric}_comparison.png"
        output_filepath = os.path.join(output_dir, filename)

        create_model_performance_barchart(
            data=plot_data,
            output_path=output_filepath,
            metric_name=metric_name,
            model_sort_order=cleaned_model_sort_order,
            algorithm_display_order=final_display_order,
            language_order=current_lang_order,
            language_palette=LANGUAGE_PALETTE,
            figsize_per_facet=figsize_per_facet
        )


if __name__ == "__main__":
    # Determine paths relative to the script's location
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    visualization_dir = os.path.dirname(current_dir) 
    project_root_dir = os.path.dirname(visualization_dir) 

    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    default_output_path = os.path.join(visualization_dir, "plots", "model_performance_barcharts")

    parser = argparse.ArgumentParser(description="Generate Model Performance Comparison Bar Charts.")
    parser.add_argument("--results-dir", type=str, default=default_results_path)
    parser.add_argument("--output-dir", type=str, default=default_output_path)
    parser.add_argument("--config-path", type=str, default=default_config_path)
    parser.add_argument("--output-filename-prefix", type=str, default="")

    args = parser.parse_args()

    print("--- Starting Standalone Model Performance Bar Chart Generation ---")

    # 1. Load Data
    df_all_data = extract_detailed_visualization_data(args.results_dir)
    if df_all_data is None or df_all_data.empty:
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load configurations
    standalone_model_sort_order = None
    try:
        if os.path.exists(args.config_path):
            config_loader = ConfigLoader(args.config_path)
            viz_settings = config_loader.config.get("visualization_settings", {})
            standalone_model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
    except Exception:
        pass

    if not standalone_model_sort_order and 'question_model' in df_all_data.columns:
        standalone_model_sort_order = sorted(df_all_data['question_model'].unique().tolist())

    # 3. Generate plots
    # Standard Algorithm Comparison
    generate_model_performance_barcharts(
        df_data=df_all_data,
        output_dir=args.output_dir,
        output_filename_prefix=args.output_filename_prefix,
        model_sort_order=standalone_model_sort_order
    )
    
    # New Format Comparison
    generate_format_comparison_barcharts(
        df_data=df_all_data,
        output_dir=args.output_dir,
        output_filename_prefix=args.output_filename_prefix,
        model_sort_order=standalone_model_sort_order
    )

    print("\n--- Standalone Model Performance Bar Chart Generation Finished ---")
