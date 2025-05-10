# visualization/plot_scripts/linechart_generators.py
import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

# Assuming plot_utils.py is in the same directory or accessible via PYTHONPATH
try:
    from .plot_utils import add_project_paths, sanitize_filename, get_model_colors
except ImportError:
    # Fallback if running from a different context or plot_utils is structured differently
    # This assumes plot_utils.py is in the same directory as this script.
    # Adjust if your project structure is different.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import add_project_paths, sanitize_filename, get_model_colors


add_project_paths() # Ensure project paths are set for other imports

from visualization.visualization_data_extractor import extract_detailed_visualization_data
from visualization.plot_scripts.linecharts import create_zeroshot_noise_level_linechart
from utils.config_loader import ConfigLoader # For loading sort order / colors in standalone

# Metrics to generate plots for
TARGET_ZEROSHOT_METRICS = ["f1_score", "accuracy", "precision", "recall", "specificity"]

def generate_zeroshot_performance_linecharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None,
    model_palette: Optional[Dict[str, str]] = None,
    languages_to_plot: Optional[List[str]] = None, # Optional: to restrict languages
    figsize: Tuple[int, int] = (12, 7)
) -> None:
    """
    Generates line charts for zero-shot performance (various metrics vs. noise level)
    for each specified language.

    Args:
        df_data: Pandas DataFrame containing all extracted visualization data.
        output_dir: Directory to save the generated plots.
        output_filename_prefix: Optional prefix for plot filenames.
        model_sort_order: List of model names to define the order of lines in plots.
        model_palette: Dictionary mapping model names to colors for consistent line colors.
        languages_to_plot: Optional list of languages to generate plots for. If None,
                           plots for all languages found in zeroshot data will be generated.
        figsize: Tuple specifying the figure size.
    """
    print("\n--- Generating Zero-shot Performance Line Charts ---")
    
    required_cols = [
        "language", "question_model", "retrieval_algorithm", 
        "noise_level", "metric_type", "metric_value"
    ]
    if not all(col in df_data.columns for col in required_cols):
        print(f"Error: Input DataFrame is missing one or more required columns: {required_cols}. Found: {df_data.columns.tolist()}")
        print("Skipping zero-shot line chart generation.")
        return

    # Filter for zero-shot data
    df_zeroshot = df_data[df_data["retrieval_algorithm"] == "zeroshot"].copy()

    if df_zeroshot.empty:
        print("No zero-shot data found. Skipping line chart generation.")
        return

    # Determine languages to process
    available_languages = sorted(df_zeroshot["language"].unique())
    if languages_to_plot:
        langs_to_process = [lang for lang in languages_to_plot if lang in available_languages]
        if not langs_to_process:
            print(f"Specified languages {languages_to_plot} not found in zero-shot data. Available: {available_languages}")
            return
    else:
        langs_to_process = available_languages

    print(f"Found zero-shot data for languages: {langs_to_process}")
    print(f"Metrics to plot: {TARGET_ZEROSHOT_METRICS}")
    if model_sort_order:
        print(f"Using model sort order: {model_sort_order}")
    if model_palette:
        print(f"Using model color palette: {list(model_palette.keys())[:5]}... colors")


    for lang in langs_to_process:
        print(f"\nProcessing language: {lang.title()}")
        df_lang_zeroshot = df_zeroshot[df_zeroshot["language"] == lang]

        if df_lang_zeroshot.empty:
            print(f"  No zero-shot data for language '{lang}'. Skipping.")
            continue

        for metric_name in TARGET_ZEROSHOT_METRICS:
            print(f"  Generating line chart for metric: {metric_name}")
            
            df_metric = df_lang_zeroshot[df_lang_zeroshot["metric_type"] == metric_name]

            if df_metric.empty:
                print(f"    No data found for metric '{metric_name}' in language '{lang}'. Skipping plot.")
                continue
            
            # Expects columns: 'noise_level', 'metric_value', 'question_model'
            # Data passed to create_zeroshot_noise_level_linechart should already be filtered.
            plot_data_for_metric = df_metric[['noise_level', 'metric_value', 'question_model']].copy()

            if plot_data_for_metric.empty or plot_data_for_metric['question_model'].nunique() == 0:
                print(f"    Data for metric '{metric_name}' in language '{lang}' is empty or has no models after filtering. Skipping plot.")
                continue

            sanitized_lang = sanitize_filename(lang)
            sanitized_metric = sanitize_filename(metric_name)
            
            filename = f"{output_filename_prefix}zeroshot_{sanitized_metric}_vs_noise_{sanitized_lang}.png"
            output_filepath = os.path.join(output_dir, filename)

            create_zeroshot_noise_level_linechart(
                data=plot_data_for_metric,
                output_path=output_filepath,
                metric_name=metric_name,
                language=lang,
                model_sort_order=model_sort_order,
                palette=model_palette,
                figsize=figsize
            )
    print("\n--- Zero-shot Performance Line Chart Generation Finished ---")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.dirname(current_dir) # ../
    project_root_dir = os.path.dirname(visualization_dir) # ../../

    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    default_output_path = os.path.join(visualization_dir, "plots", "zeroshot_linecharts")

    parser = argparse.ArgumentParser(description="Generate Zero-shot Performance Line Charts.")
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
        help="Path to the configuration JSON file (for model sort order and colors)."
    )
    parser.add_argument(
        "--output-filename-prefix", type=str, default="",
        help="Optional prefix for generated plot filenames."
    )
    parser.add_argument(
        "--languages", type=str, nargs="+", default=None,
        help="Optional: Specific languages to generate plots for (e.g., english german)."
    )

    args = parser.parse_args()

    print("--- Starting Standalone Zero-shot Line Chart Generation ---")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config path: {args.config_path}")
    if args.output_filename_prefix:
        print(f"Filename prefix: {args.output_filename_prefix}")
    if args.languages:
        print(f"Target languages: {args.languages}")


    # 1. Load Data
    print(f"\nExtracting detailed data from: {args.results_dir}")
    df_all_data = extract_detailed_visualization_data(args.results_dir)

    if df_all_data is None or df_all_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load configurations (model sort order, colors)
    standalone_model_sort_order = None
    standalone_model_palette = None
    try:
        if os.path.exists(args.config_path):
            config_loader = ConfigLoader(args.config_path)
            # Assuming REPORT_MODEL_SORT_ORDER is a key in config.json or a sub-object
            # For this example, let's assume it's under a "visualization_settings" key
            viz_settings = config_loader.config.get("visualization_settings", {})
            standalone_model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
            
            # For model colors, use the get_model_colors utility if available,
            # or try to load a map from config.
            # This assumes get_model_colors can take a ConfigLoader object or path.
            # For simplicity, let's try to get a direct map or use the utility.
            # standalone_model_palette = viz_settings.get("MODEL_COLOR_MAP")
            # if not standalone_model_palette:
            # Use the utility with all known models from the data
            all_models_in_data = df_all_data["question_model"].unique().tolist()
            standalone_model_palette = get_model_colors(all_models_in_data, config_loader.config) # Pass full config

            if standalone_model_sort_order:
                print(f"Loaded model sort order from config: {standalone_model_sort_order}")
            else:
                print("Warning: REPORT_MODEL_SORT_ORDER not found in config. Using default model order (alphabetical).")
            
            if standalone_model_palette:
                 print(f"Loaded model color palette for {len(standalone_model_palette)} models.")
            else:
                print("Warning: MODEL_COLOR_MAP not found or generated. Seaborn default palette will be used.")

        else:
            print(f"Warning: Config file not found at '{args.config_path}'. Using default model order and colors.")
    except Exception as e:
        print(f"Warning: Error loading config file '{args.config_path}': {e}. Using defaults.")
        # Provide a fallback sort order for testing if config fails
        if standalone_model_sort_order is None and 'question_model' in df_all_data.columns:
            standalone_model_sort_order = sorted(df_all_data['question_model'].unique().tolist())
            print(f"Using fallback alphabetical model sort order: {standalone_model_sort_order[:5]}...")


    # 3. Generate plots
    generate_zeroshot_performance_linecharts(
        df_data=df_all_data,
        output_dir=args.output_dir,
        output_filename_prefix=args.output_filename_prefix,
        model_sort_order=standalone_model_sort_order,
        model_palette=standalone_model_palette,
        languages_to_plot=args.languages
    )

    print("\n--- Standalone Zero-shot Line Chart Generation Finished ---")
    print(f"Plots saved to: {args.output_dir}")