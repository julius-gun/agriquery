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
    from heatmap_generators import ( # Changed import path: removed 'plot_scripts.'
        generate_global_overview_heatmaps,
        generate_english_format_heatmaps,
        generate_markdown_overview_heatmaps
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    # Setup paths
    # Output now goes to visualization/plots inside project root
    default_results_path = os.path.join(PROJECT_ROOT, "results")
    # Simplified default_output_path calculation and removed visualization_dir variable
    default_output_path = os.path.join(PROJECT_ROOT, "visualization", "plots")

    parser = argparse.ArgumentParser(description="Generate RAG Heatmaps.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=default_results_path,
        help="Directory containing the JSON result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_path,
        help="Directory where the generated plot images will be saved.",
    )

    args = parser.parse_args()

    print("--- Starting Visualization Generation ---")
    print(f"Results Directory: {args.results_dir}")
    print(f"Output Directory: {args.output_dir}")

    # 1. Extract Data
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted.")
        return

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Generate Global Overview Heatmaps
    generate_global_overview_heatmaps(df_data, args.output_dir)

    # 3. Generate English Format Heatmaps
    generate_english_format_heatmaps(df_data, args.output_dir)

    # 4. Generate Markdown Overview Heatmaps
    generate_markdown_overview_heatmaps(df_data, args.output_dir)

    print("\n--- Visualization Generation Finished ---")

if __name__ == "__main__":
    main()
