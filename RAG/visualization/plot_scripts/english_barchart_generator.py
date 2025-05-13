import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Tuple

# Assuming plot_utils.py and the new english_barcharts.py are accessible
try:
    # Use . for relative imports if this file is in the same package/directory
    from .plot_utils import add_project_paths, sanitize_filename
    from .english_barcharts import create_english_retrieval_barchart
except ImportError:
    # Fallback for standalone execution or if structure is different
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import add_project_paths, sanitize_filename
    from english_barcharts import create_english_retrieval_barchart

add_project_paths()

from visualization.visualization_data_extractor import extract_detailed_visualization_data
from utils.config_loader import ConfigLoader

# --- Configuration for English Bar Charts ---
TARGET_LANGUAGE = "english"
TARGET_METRICS_FOR_ENGLISH_BARCHART = ["accuracy", "f1_score"]

# Retrieval methods to include and their internal names
# Display names will be used as is from this dict for RAG methods
RETRIEVAL_METHODS_TO_PLOT = {
    "hybrid": "Hybrid",
    "embedding": "Embedding",
    "keyword": "Keyword",
}

# Configuration for the "Full Manual" data point (zeroshot at specific noise level)
FULL_MANUAL_ALIAS = "Full Manual"
FULL_MANUAL_NOISE_LEVEL = 59000 # Corresponds to zeroshot at this noise level

# Desired order of retrieval methods on the plot
RETRIEVAL_METHOD_DISPLAY_ORDER = [
    "Hybrid",
    "Embedding",
    "Keyword",
    FULL_MANUAL_ALIAS
]

# Specific model name mappings for beautification (mirrors barchart_generators.py)
MODEL_NAME_MAPPINGS = {
    "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
    "phi3_14B_q4_medium-128k": "phi3 14B",
    # Add other specific mappings here if needed
}

def clean_model_name(model_name: str) -> str:
    """
    Applies specific cleaning rules to a model name.
    Mirrors the version in barchart_generators.py for consistency.
    """
    cleaned_name = MODEL_NAME_MAPPINGS.get(model_name, model_name)
    if cleaned_name == model_name:
        if cleaned_name.endswith("-128k"):
            cleaned_name = cleaned_name.removesuffix("-128k")
        cleaned_name = cleaned_name.replace("_", " ")
    return cleaned_name

def generate_english_retrieval_comparison_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (18, 9) # Adjusted default figsize
) -> None:
    """
    Generates bar charts for English results comparing Accuracy and F1 Score
    across specified retrieval methods for different models.

    Args:
        df_data: Pandas DataFrame containing all extracted visualization data.
        output_dir: Directory to save the generated plots.
        output_filename_prefix: Optional prefix for plot filenames.
        model_sort_order: List of model names for x-axis order.
        figsize: Figure size for the plots.
    """
    print("\n--- Generating English Retrieval Comparison Bar Charts ---")

    required_cols = [
        "language", "question_model", "retrieval_algorithm", "noise_level",
        "metric_type", "metric_value"
    ]
    if not all(col in df_data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_data.columns]
        print(f"Error: Input DataFrame is missing required columns: {missing_cols}. Found: {df_data.columns.tolist()}")
        print("Skipping English retrieval comparison bar chart generation.")
        return

    # 0. Apply model name cleaning
    df_plot_base = df_data.copy()
    df_plot_base['question_model'] = df_plot_base['question_model'].apply(clean_model_name)
    print(f"Applied model name cleaning. Unique models after cleaning: {df_plot_base['question_model'].nunique()}")

    cleaned_model_sort_order = None
    if model_sort_order:
        cleaned_model_sort_order = [clean_model_name(model) for model in model_sort_order]
        print(f"Cleaned model sort order: {cleaned_model_sort_order}")


    # 1. Filter for English language results
    df_english = df_plot_base[df_plot_base["language"] == TARGET_LANGUAGE].copy()
    if df_english.empty:
        print(f"No data found for language '{TARGET_LANGUAGE}'. Skipping charts.")
        return
    print(f"Found {len(df_english)} data points for '{TARGET_LANGUAGE}' language.")

    # 2. Prepare "Full Manual" data
    df_full_manual = df_english[
        (df_english["retrieval_algorithm"] == "zeroshot") &
        (df_english["noise_level"] == FULL_MANUAL_NOISE_LEVEL)
    ].copy()

    if df_full_manual.empty:
        print(f"Warning: No '{TARGET_LANGUAGE}' data found for 'zeroshot' at noise level {FULL_MANUAL_NOISE_LEVEL}. "
              f"'{FULL_MANUAL_ALIAS}' will be missing from plots.")
    else:
        df_full_manual["retrieval_method_display"] = FULL_MANUAL_ALIAS
        print(f"Found {len(df_full_manual)} data points for '{FULL_MANUAL_ALIAS}'. "
              f"Models: {df_full_manual['question_model'].unique().tolist()}")

    # 3. Prepare RAG data (Hybrid, Embedding, Keyword)
    rag_data_frames = []
    present_retrieval_methods_for_plot = []

    for algo_internal_name, algo_display_name in RETRIEVAL_METHODS_TO_PLOT.items():
        df_rag_algo = df_english[df_english["retrieval_algorithm"] == algo_internal_name].copy()
        if not df_rag_algo.empty:
            df_rag_algo["retrieval_method_display"] = algo_display_name
            rag_data_frames.append(df_rag_algo)
            present_retrieval_methods_for_plot.append(algo_display_name)
            print(f"Found {len(df_rag_algo)} data points for RAG method: '{algo_display_name}' (internal: '{algo_internal_name}')")
            print(f"  Models for '{algo_display_name}': {df_rag_algo['question_model'].unique().tolist()}")
        else:
            print(f"Warning: No '{TARGET_LANGUAGE}' data found for RAG method '{algo_internal_name}'. "
                  f"'{algo_display_name}' will be missing from plots.")

    # Combine all data frames
    all_plot_data_frames = rag_data_frames
    if not df_full_manual.empty:
        all_plot_data_frames.append(df_full_manual)

    if not all_plot_data_frames:
        print("No data to plot after filtering for English and specified retrieval methods. Skipping.")
        return

    df_combined_english = pd.concat(all_plot_data_frames, ignore_index=True)
    if df_combined_english.empty:
        print("Combined English data is empty. Skipping chart generation.")
        return

    # Filter cleaned_model_sort_order to include only models present in the combined English data
    if cleaned_model_sort_order:
        models_in_combined_data = df_combined_english['question_model'].unique().tolist()
        final_model_sort_order = [m for m in cleaned_model_sort_order if m in models_in_combined_data]
        if not final_model_sort_order and models_in_combined_data: # If sort order made it empty but data exists
             print(f"Warning: None of the models in `model_sort_order` are present in the filtered English data. Plotting available models alphabetically.")
             final_model_sort_order = sorted(models_in_combined_data)
        elif not models_in_combined_data: # No models in data at all
            print("Error: No models found in the combined English data to plot.")
            return
    else: # Default to alphabetical if no sort order provided
        final_model_sort_order = sorted(df_combined_english['question_model'].unique().tolist())
    
    print(f"Final model sort order for plots: {final_model_sort_order}")

    # Filter retrieval_method_display_order to include only methods present in the combined data
    actual_methods_in_data = df_combined_english['retrieval_method_display'].unique().tolist()
    final_retrieval_method_order = [m for m in RETRIEVAL_METHOD_DISPLAY_ORDER if m in actual_methods_in_data]
    # Add any other methods found in data that weren't in the predefined order (shouldn't happen with current setup)
    for method in actual_methods_in_data:
        if method not in final_retrieval_method_order:
            final_retrieval_method_order.append(method) # Add to end
    
    if not final_retrieval_method_order:
        print("Error: No retrieval methods found in the combined English data to plot.")
        return
    print(f"Final retrieval method order for plots: {final_retrieval_method_order}")


    # 4. Generate plot for each target metric
    for metric_name in TARGET_METRICS_FOR_ENGLISH_BARCHART:
        print(f"\n  Generating English barchart for metric: {metric_name}")

        df_metric_specific = df_combined_english[df_combined_english["metric_type"] == metric_name].copy()

        if df_metric_specific.empty:
            print(f"    No '{TARGET_LANGUAGE}' data found for metric '{metric_name}' with selected retrieval methods. Skipping this plot.")
            continue

        # Data for plotting needs: 'question_model', 'metric_value', 'retrieval_method_display'
        plot_data_for_metric = df_metric_specific[[
            "question_model", "metric_value", "retrieval_method_display"
        ]].copy()

        if plot_data_for_metric.empty or plot_data_for_metric['question_model'].nunique() == 0:
            print(f"    Data for metric '{metric_name}' is empty or has no models after final selection. Skipping plot.")
            continue
        
        print(f"    Plotting {len(plot_data_for_metric)} data points for {metric_name}.")
        print(f"    Models in this metric's data: {plot_data_for_metric['question_model'].unique().tolist()}")
        print(f"    Retrieval methods in this metric's data: {plot_data_for_metric['retrieval_method_display'].unique().tolist()}")


        sanitized_metric = sanitize_filename(metric_name)
        filename = f"{output_filename_prefix}english_retrieval_{sanitized_metric}_comparison.png"
        output_filepath = os.path.join(output_dir, filename)

        create_english_retrieval_barchart(
            data=plot_data_for_metric,
            output_path=output_filepath,
            metric_name=metric_name,
            model_sort_order=final_model_sort_order,
            retrieval_method_order=final_retrieval_method_order,
            figsize=figsize
            # Other styling args like font sizes can be passed if needed
        )
    print("\n--- English Retrieval Comparison Bar Chart Generation Finished ---")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(visualization_dir)

    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    # Default output to a subfolder for these specific charts
    default_output_path = os.path.join(visualization_dir, "plots", "english_retrieval_barcharts")

    parser = argparse.ArgumentParser(description="Generate English Retrieval Comparison Bar Charts.")
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
    parser.add_argument(
        "--figsize-width", type=float, default=18, help="Width of the figure for plots."
    )
    parser.add_argument(
        "--figsize-height", type=float, default=9, help="Height of the figure for plots."
    )

    args = parser.parse_args()

    print("--- Starting Standalone English Retrieval Bar Chart Generation ---")
    print(f"Results Dir: {args.results_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Config Path: {args.config_path}")
    print(f"Filename Prefix: '{args.output_filename_prefix}'")
    print(f"Figure Size: ({args.figsize_width}, {args.figsize_height})")


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
            # Assuming the same sort order is relevant here
            standalone_model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
            if standalone_model_sort_order:
                print(f"Loaded model sort order from config: {standalone_model_sort_order}")
            else:
                print(f"Warning: REPORT_MODEL_SORT_ORDER not found in config at '{args.config_path}'. Plots will use alphabetical model order if no models are filtered out.")
        else:
            print(f"Warning: Config file not found at '{args.config_path}'. Plots will use alphabetical model order.")
    except Exception as e:
        print(f"Warning: Error loading config file '{args.config_path}': {e}.")

    # If no sort order loaded, it will default to alphabetical within the generator function
    # after data filtering.

    # 3. Generate plots
    generate_english_retrieval_comparison_barcharts(
        df_data=df_all_data,
        output_dir=args.output_dir,
        output_filename_prefix=args.output_filename_prefix,
        model_sort_order=standalone_model_sort_order,
        figsize=(args.figsize_width, args.figsize_height)
    )

    print("\n--- Standalone English Retrieval Bar Chart Generation Finished ---")
    print(f"Plots saved to: {args.output_dir}")