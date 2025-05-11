# visualization/plot_scripts/barchart_generators.py
import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Tuple

# Assuming plot_utils.py and barcharts.py are in the same directory or accessible
try:
    from .plot_utils import add_project_paths, sanitize_filename, get_model_colors
    from .barcharts import create_model_performance_barchart, METRIC_DISPLAY_NAMES, LANGUAGE_ORDER, LANGUAGE_PALETTE
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import add_project_paths, sanitize_filename, get_model_colors
    from barcharts import create_model_performance_barchart, METRIC_DISPLAY_NAMES, LANGUAGE_ORDER, LANGUAGE_PALETTE

add_project_paths()

from visualization.visualization_data_extractor import extract_detailed_visualization_data
from utils.config_loader import ConfigLoader

# Metrics to generate bar charts for
TARGET_METRICS_FOR_BARCHART = ["f1_score", "accuracy", "precision", "recall", "specificity"]

# RAG algorithms to include in the comparison.
# These should match the 'retrieval_algorithm' values in the data.
# Display names can be customized if needed.
RAG_ALGORITHMS_TO_PLOT = {
    "hybrid": "Hybrid RAG",
    "keyword": "Keyword RAG",
    "embedding": "Embedding RAG",
    # Add other RAG algorithms here if they exist and should be plotted
}

# Configuration for the "Full Manual" data point
FULL_MANUAL_ALIAS = "Full Manual 59k tokens"
FULL_MANUAL_NOISE_LEVEL = 59000 # zeroshot at this noise level

# Order in which algorithm facets should appear
# The generator will try to match these from RAG_ALGORITHMS_TO_PLOT values and FULL_MANUAL_ALIAS
DEFAULT_ALGORITHM_DISPLAY_ORDER = ["Hybrid", "Embedding", "Keyword", FULL_MANUAL_ALIAS]


def generate_model_performance_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None,
    
    
    # language_order and palette are taken from barcharts.py defaults for now
    # but could be passed if customization is needed at this level.
    # languages_to_plot: Optional[List[str]] = None, # If specific languages are needed
    figsize_per_facet: Tuple[float, float] = (7, 5.5)
) -> None:
    """
    Generates grouped bar charts comparing model performance (various metrics)
    for selected RAG algorithms and "Full Manual" (zeroshot 59k tokens).

    Args:
        df_data: Pandas DataFrame containing all extracted visualization data.
        output_dir: Directory to save the generated plots.
        output_filename_prefix: Optional prefix for plot filenames.
        model_sort_order: List of model names for x-axis order.
        figsize_per_facet: Size of each facet in the plot.
    """
    print("\n--- Generating Model Performance Bar Charts ---")

    required_cols = [
        "language", "question_model", "retrieval_algorithm", "noise_level",
        "metric_type", "metric_value"
    ]
    if not all(col in df_data.columns for col in required_cols):
        print(f"Error: Input DataFrame is missing one or more required columns: {required_cols}. Found: {df_data.columns.tolist()}")
        print("Skipping model performance bar chart generation.")
        return

    df_plot_base = df_data.copy()

    # 1. Prepare "Full Manual" data
    df_full_manual = df_plot_base[
        (df_plot_base["retrieval_algorithm"] == "zeroshot") &
        (df_plot_base["noise_level"] == FULL_MANUAL_NOISE_LEVEL)
    ].copy()

    if df_full_manual.empty:
        print(f"Warning: No data found for 'zeroshot' at noise level {FULL_MANUAL_NOISE_LEVEL}. '{FULL_MANUAL_ALIAS}' will be missing from plots.")
        # df_full_manual will remain empty, subsequent concatenation won't add it.
    else:
        df_full_manual["retrieval_algorithm_display"] = FULL_MANUAL_ALIAS
        print(f"Found {len(df_full_manual['question_model'].unique())} models for '{FULL_MANUAL_ALIAS}' data across {len(df_full_manual['language'].unique())} languages.")

    # 2. Prepare RAG data
    rag_data_frames = []
    present_rag_algorithms_for_plot = []
    for algo_internal_name, algo_display_name in RAG_ALGORITHMS_TO_PLOT.items():
        df_rag_algo = df_plot_base[df_plot_base["retrieval_algorithm"] == algo_internal_name].copy()
        if not df_rag_algo.empty:
            df_rag_algo["retrieval_algorithm_display"] = algo_display_name
            rag_data_frames.append(df_rag_algo)
            present_rag_algorithms_for_plot.append(algo_display_name)
            print(f"Found data for RAG algorithm: '{algo_display_name}' (internal: '{algo_internal_name}')")
        else:
            print(f"Warning: No data found for RAG algorithm '{algo_internal_name}'. It will be missing from plots.")
    
    # Determine the actual order of algorithm facets based on available data and desired order
    final_algorithm_display_order = [
        algo for algo in DEFAULT_ALGORITHM_DISPLAY_ORDER 
        if algo == FULL_MANUAL_ALIAS or algo in present_rag_algorithms_for_plot
    ]
    # Add any other found RAG algorithms not in default order, alphabetically
    for algo_disp_name in sorted(present_rag_algorithms_for_plot):
        if algo_disp_name not in final_algorithm_display_order:
            final_algorithm_display_order.append(algo_disp_name)
    
    # Add Full Manual if it exists and not already included by some chance
    if not df_full_manual.empty and FULL_MANUAL_ALIAS not in final_algorithm_display_order:
        # This case might be rare if FULL_MANUAL_ALIAS is in DEFAULT_ALGORITHM_DISPLAY_ORDER
        final_algorithm_display_order.append(FULL_MANUAL_ALIAS)
        
    if not rag_data_frames and df_full_manual.empty:
        print("No RAG data or 'Full Manual' data found. Skipping bar chart generation.")
        return

    # 3. Combine data
    all_plot_data_frames = rag_data_frames
    if not df_full_manual.empty:
        all_plot_data_frames.append(df_full_manual)

    if not all_plot_data_frames:
        print("No data to plot after filtering for RAG and Full Manual. Skipping.")
        return

    df_combined = pd.concat(all_plot_data_frames, ignore_index=True)

    if df_combined.empty:
        print("Combined data is empty. Skipping bar chart generation.")
        return
    
    print(f"Models to plot: {model_sort_order if model_sort_order else 'Default (alphabetical)'}")
    print(f"Algorithm facets to plot (in order): {final_algorithm_display_order}")
    print(f"Languages to plot (hue order): {LANGUAGE_ORDER}")

    for metric_name in TARGET_METRICS_FOR_BARCHART:
        print(f"\n  Generating bar chart for metric: {metric_name}")

        df_metric_specific = df_combined[df_combined["metric_type"] == metric_name].copy()

        if df_metric_specific.empty:
            print(f"    No data found for metric '{metric_name}'. Skipping this plot.")
            continue
        
        # Ensure only languages present in the data for this metric are in language_order for the plot
        # (though barcharts.py also handles this, good to be aware)
        actual_languages_in_metric_data = sorted(df_metric_specific['language'].unique())
        current_lang_order = [lang for lang in LANGUAGE_ORDER if lang in actual_languages_in_metric_data]
        for lang in actual_languages_in_metric_data: # Add any new languages from data
            if lang not in current_lang_order:
                current_lang_order.append(lang)

        # The plotting function expects 'question_model', 'metric_value', 'language', 'retrieval_algorithm_display'
        plot_data_for_metric = df_metric_specific[[
            "question_model", "metric_value", "language", "retrieval_algorithm_display"
        ]].copy()

        if plot_data_for_metric.empty or plot_data_for_metric['question_model'].nunique() == 0:
            print(f"    Data for metric '{metric_name}' is empty or has no models after final selection. Skipping plot.")
            continue

        sanitized_metric = sanitize_filename(metric_name)
        filename = f"{output_filename_prefix}model_perf_{sanitized_metric}_comparison.png"
        output_filepath = os.path.join(output_dir, filename)

        create_model_performance_barchart(
            data=plot_data_for_metric,
            output_path=output_filepath,
            metric_name=metric_name,
            model_sort_order=model_sort_order,
            algorithm_display_order=final_algorithm_display_order,
            language_order=current_lang_order, # Use filtered and ordered list
            language_palette=LANGUAGE_PALETTE, # From barcharts.py
            figsize=figsize_per_facet
        )
    print("\n--- Model Performance Bar Chart Generation Finished ---")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(visualization_dir)

    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    default_output_path = os.path.join(visualization_dir, "plots", "model_performance_barcharts")

    parser = argparse.ArgumentParser(description="Generate Model Performance Comparison Bar Charts.")
    parser.add_argument(
        "--results-dir", type=str, default=default_results_path,
        help="Directory containing the JSON result files."
    )
    parser.add_argument(
        "--output-dir", type=str, default=default_output_path,
        help="Directory where the generated plot images will be saved."
    )
    parser.add_argument(
        "--config-path", type=str, default=default_config_path,
        help="Path to the configuration JSON file (for model sort order)."
    )
    parser.add_argument(
        "--output-filename-prefix", type=str, default="",
        help="Optional prefix for generated plot filenames."
    )
    # Potentially add --languages argument if needed later

    args = parser.parse_args()

    print("--- Starting Standalone Model Performance Bar Chart Generation ---")
    # ... (print args similar to linechart_generators.py) ...

    # 1. Load Data
    df_all_data = extract_detailed_visualization_data(args.results_dir)
    if df_all_data is None or df_all_data.empty:
        print("Exiting: No data extracted.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load configurations (model sort order)
    standalone_model_sort_order = None
    try:
        if os.path.exists(args.config_path):
            config_loader = ConfigLoader(args.config_path)
            viz_settings = config_loader.config.get("visualization_settings", {})
            standalone_model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
            if standalone_model_sort_order:
                print(f"Loaded model sort order from config: {standalone_model_sort_order}")
            else:
                print("Warning: REPORT_MODEL_SORT_ORDER not found in config.")
        else:
            print(f"Warning: Config file not found at '{args.config_path}'.")
    except Exception as e:
        print(f"Warning: Error loading config file '{args.config_path}': {e}.")

    if not standalone_model_sort_order and 'question_model' in df_all_data.columns:
        standalone_model_sort_order = sorted(df_all_data['question_model'].unique().tolist())
        print(f"Using fallback alphabetical model sort order: {standalone_model_sort_order[:5]}...")


    # 3. Generate plots
    generate_model_performance_barcharts(
        df_data=df_all_data,
        output_dir=args.output_dir,
        output_filename_prefix=args.output_filename_prefix,
        model_sort_order=standalone_model_sort_order
        # figsize_per_facet can be adjusted here if needed for standalone runs
    )

    print("\n--- Standalone Model Performance Bar Chart Generation Finished ---")
    print(f"Plots saved to: {args.output_dir}")