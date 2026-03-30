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
        generate_format_comparison_heatmaps
    )
    from barchart_generators import (
        generate_model_performance_barcharts,
        generate_format_comparison_barcharts,
        generate_token_efficiency_barchart
    )
    # Added new generators
    from scatter_plot_generators import (
        generate_cross_lingual_scatter_plots,
        generate_model_efficiency_plot,
        generate_performance_gap_plot
    )
    from latex_table_generator import generate_latex_report
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    # Setup paths
    default_results_path = os.path.join(PROJECT_ROOT, "results")
    default_output_path = os.path.join(PROJECT_ROOT, "visualization", "plots")
    default_table_path = os.path.join(PROJECT_ROOT, "visualization", "latex_tables")
    default_config_path = os.path.join(PROJECT_ROOT, "config.json")

    parser = argparse.ArgumentParser(description="Generate RAG Visualizations for Paper.")
    parser.add_argument("--results-dir", type=str, default=default_results_path)
    parser.add_argument("--output-dir", type=str, default=default_output_path)
    parser.add_argument("--table-dir", type=str, default=default_table_path)
    parser.add_argument("--config-path", type=str, default=default_config_path)

    args = parser.parse_args()

    print("--- Starting Paper Visualization Generation ---")

    # 1. Extract Data
    df_data = extract_detailed_visualization_data(args.results_dir)
    if df_data is None or df_data.empty:
        print("Warning: No results data extracted. Skipping result plots.")
        # We continue because we might still want the token efficiency plot
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    # Load Model Sort Order
    model_sort_order = None
    try:
        config_loader = ConfigLoader(args.config_path)
        viz_settings = config_loader.config.get("visualization_settings", {})
        model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
    except Exception:
        pass

    if df_data is not None and not df_data.empty:
        # 2. Heatmaps
        generate_global_overview_heatmaps(df_data, args.output_dir, model_sort_order)
        generate_format_comparison_heatmaps(df_data, args.output_dir, model_sort_order)

        # 3. Bar Charts (Performance)
        barchart_dir = os.path.join(args.output_dir, "barcharts")
        generate_model_performance_barcharts(df_data, barchart_dir, model_sort_order=model_sort_order)
        generate_format_comparison_barcharts(df_data, barchart_dir, model_sort_order=model_sort_order)

        # 4. Scatter / Line Plots
        # Existing Cross-Lingual Scatter
        scatter_dir = os.path.join(args.output_dir, "scatterplots")
        generate_cross_lingual_scatter_plots(df_data, scatter_dir, model_sort_order=model_sort_order)
        
        # New: Model Size vs Performance
        generate_model_efficiency_plot(df_data, args.output_dir) # Saves to root output dir as per prompt
        
        # New: Performance Gap (The Cross-Lingual Tax)
        generate_performance_gap_plot(df_data, args.output_dir)

        # 5. LaTeX Tables
        generate_latex_report(df_data, args.table_dir, model_sort_order)

    # 6. Token Efficiency Chart (Format Comparison)
    try:
    
        print("\n--- Calculating Token Counts for Efficiency Chart ---")
        # Ensure analysis module is in path (it is child of PROJECT_ROOT)
        sys.path.append(str(PROJECT_ROOT))
        from analysis.token_counter import TokenCounter
        
        # Initialize counter (uses configuration to find manuals)
        counter = TokenCounter(config_path=args.config_path)
        token_results = counter.count_tokens_for_all_manuals()
        
        # # Generate Plot
        # generate_token_efficiency_barchart(token_results, args.output_dir)
        
        # Generate LaTeX Table
        counter.save_latex_table(token_results, args.table_dir)
        
    except ImportError as e:
        print(f"Skipping Token Efficiency Chart: Could not import TokenCounter ({e})")
    except Exception as e:
        print(f"Skipping Token Efficiency Chart: Error ({e})")

    print("\n--- Visualization Generation Finished ---")

if __name__ == "__main__":
    main()
