import os
import sys
import argparse

# --- Path Setup ---
try:
    from plot_utils import add_project_paths
    PROJECT_ROOT = add_project_paths()
except ImportError as e:
    print(f"Error importing plot_utils: {e}")
    sys.exit(1)

try:
    from visualization_data_extractor import extract_detailed_visualization_data
    from heatmap_generators import (
        generate_global_overview_heatmaps,
        generate_english_format_heatmaps,
        generate_format_comparison_heatmaps,
        generate_markdown_overview_heatmaps
    )
    from barchart_generators import (
        generate_model_performance_barcharts,
        generate_format_comparison_barcharts
    )
    from english_barchart_generator import generate_english_retrieval_comparison_barcharts
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    # Setup paths
    default_results_path = os.path.join(PROJECT_ROOT, "results")
    default_output_path = os.path.join(PROJECT_ROOT, "visualization", "plots")
    default_config_path = os.path.join(PROJECT_ROOT, "config.json")

    parser = argparse.ArgumentParser(description="Generate RAG Visualizations.")
    parser.add_argument("--results-dir", type=str, default=default_results_path)
    parser.add_argument("--output-dir", type=str, default=default_output_path)
    parser.add_argument("--config-path", type=str, default=default_config_path)

    args = parser.parse_args()

    print("--- Starting Visualization Generation ---")
    print(f"Results Directory: {args.results_dir}")
    print(f"Output Directory: {args.output_dir}")

    # 1. Extract Data
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model Sort Order from Config
    model_sort_order = None
    try:
        config_loader = ConfigLoader(args.config_path)
        viz_settings = config_loader.config.get("visualization_settings", {})
        model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
        if model_sort_order:
            print(f"Loaded model sort order: {len(model_sort_order)} models defined.")
    except Exception as e:
        print(f"Warning: Could not load config for sort order: {e}")

    # 2. Heatmaps
    generate_global_overview_heatmaps(df_data, args.output_dir)
    generate_english_format_heatmaps(df_data, args.output_dir)
    generate_format_comparison_heatmaps(df_data, args.output_dir)
    generate_markdown_overview_heatmaps(df_data, args.output_dir)

    # 3. Bar Charts
    # Original Algorithm Comparison (Filtered to Markdown internally for clean comparison)
    generate_model_performance_barcharts(
        df_data, 
        os.path.join(args.output_dir, "model_performance_barcharts"), 
        model_sort_order=model_sort_order
    )
    # New Format Comparison
    generate_format_comparison_barcharts(
        df_data, 
        os.path.join(args.output_dir, "model_performance_barcharts"), 
        model_sort_order=model_sort_order
    )
    # English specific charts
    generate_english_retrieval_comparison_barcharts(
        df_data,
        os.path.join(args.output_dir, "english_retrieval_barcharts"),
        model_sort_order=model_sort_order
    )

    print("\n--- Visualization Generation Finished ---")

if __name__ == "__main__":
    main()
