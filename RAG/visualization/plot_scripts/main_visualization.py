# visualization/plot_scripts/main_visualization.py

# python visualization\plot_scripts\main_visualization.py --plot-type=heatmap

import os
import sys
import argparse
import pandas as pd

# --- Adjust Python Path (using the helper) ---
# Import the function and call it explicitly to ensure paths are set
# before other project imports are attempted.
try:
    # This import will also execute the path setup code in plot_utils
    # because it runs at the module level there.
    from plot_utils import add_project_paths, sanitize_filename
    PROJECT_ROOT = add_project_paths() # Ensure paths are set and get the root
except ImportError as e:
     print("Failed to import 'add_project_paths' or 'sanitize_filename' from 'plot_utils.py'.")
     print(f"Make sure 'plot_utils.py' exists in the same directory as this script ({os.path.dirname(__file__)}) and is correctly structured.")
     print(f"Original error: {e}")
     # Attempt to provide more context on where it's looking
     print(f"Current sys.path: {sys.path}")
     # Try calculating expected path for plot_utils
     expected_plot_utils_path = os.path.join(os.path.dirname(__file__), 'plot_utils.py')
     print(f"Checking for plot_utils at: {expected_plot_utils_path}")
     if not os.path.exists(expected_plot_utils_path):
         print("Error: plot_utils.py not found at the expected location.")
     sys.exit(1)


# --- Imports ---
try:
    from utils.config_loader import ConfigLoader
    from visualization.visualization_data_extractor import (
        extract_detailed_visualization_data,
    )
    from visualization.plot_scripts.boxplot_generators import (
        generate_f1_boxplot,
        generate_dataset_success_boxplot,
    )

    # Import the new heatmap generator functions
    from visualization.plot_scripts.heatmap_generators import (
        generate_language_vs_model_heatmap,
        generate_chunk_vs_overlap_heatmap,
        generate_model_vs_chunk_overlap_heatmap,
        generate_dataset_success_heatmaps,
        generate_algo_vs_model_f1_heatmap,
        generate_algo_vs_model_dataset_success_heatmap,
    )

    # Removed direct imports from heatmaps.py

except ImportError as e:
    print("Error importing required modules after attempting path setup.")
    print("This might indicate an issue with the project structure or missing files within 'utils' or 'visualization'.")
    print(f"Project root added to sys.path: {PROJECT_ROOT if 'PROJECT_ROOT' in locals() else 'Unknown'}")
    print(f"Current sys.path: {sys.path}")
    print(f"Original Error: {e}")
    sys.exit(1)

# sanitize_filename function is now imported from plot_utils


def main():
    # Get project root and visualization dir using the established PROJECT_ROOT
    visualization_dir = os.path.join(PROJECT_ROOT, "visualization") # Derived from root
    project_root_dir = PROJECT_ROOT # Use the established root

    # [...] rest of the main function (paths should now use project_root_dir)

    parser = argparse.ArgumentParser(
        description="Generate visualizations for RAG test results."
    )
    # Define default paths relative to the project structure
    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    default_output_path = os.path.join(visualization_dir, "plots")

    parser.add_argument(
        "--config-path",
        type=str,
        default=default_config_path,
        help="Path to the configuration JSON file (to get language list).",
    )
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
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        # --- MODIFICATION START: Added new choices ---
        choices=[
            "boxplot",              # F1 Boxplot (grouped)
            "heatmap",              # Original set of detailed heatmaps
            "dataset_boxplot",      # Dataset Success Boxplot (specific params)
            "algo_vs_model_f1",     # NEW: Algo vs Model F1 Heatmap
            "algo_vs_model_success",# NEW: Algo vs Model Mean Success Heatmap
            "all"                   # Generate all plot types
            ],
        # --- MODIFICATION END ---
        help="Type of plot(s) to generate. 'heatmap' generates detailed Lang/Chunk/Model heatmaps. 'algo_vs_model...' generate summary heatmaps. 'all' generates everything.",
    )
    parser.add_argument(
        "--group-by",  # Only used by F1 boxplot now
        type=str,
        default="question_model",
        choices=[
            "retrieval_algorithm",
            "language",
            "question_model",
            "chunk_size",
            "overlap_size",
            "num_retrieved_docs",
        ],
        help="Parameter to group the F1 box plots by (used if plot-type includes 'boxplot').",
    )
    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="",
        help="Optional prefix for all generated plot filenames.",
    )

    # --- Constants for dataset_boxplot ---
    DATASET_BOXPLOT_LANG = "english"
    DATASET_BOXPLOT_CHUNK = 200
    DATASET_BOXPLOT_OVERLAP = 100

    args = parser.parse_args()

    print("--- Starting Visualization Generation ---")
    print(f"Project Root: {project_root_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Requested plot type(s): {args.plot_type}")
    if args.plot_type == "dataset_boxplot" or args.plot_type == "all":
        print(
            f"Dataset Boxplot Params: Lang={DATASET_BOXPLOT_LANG}, Chunk={DATASET_BOXPLOT_CHUNK}, Overlap={DATASET_BOXPLOT_OVERLAP}"
        )
    if args.output_filename_prefix:
        print(f"Filename prefix: {args.output_filename_prefix}")

    # 0. Load Config to get languages
    all_languages_list = None
    try:
        config_loader = ConfigLoader(args.config_path)
        language_configs = config_loader.config.get("language_configs", [])
        loaded_languages = [
            lc.get("language") for lc in language_configs if lc.get("language")
        ]
        if loaded_languages:
            all_languages_list = loaded_languages
            print(
                f"Found languages in config for consistent plot elements: {all_languages_list}"
            )
        else:
            print(
                f"Warning: No languages found in 'language_configs' in {args.config_path}. Plot elements order may vary."
            )
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at '{args.config_path}'. Cannot determine full language list for consistent plot elements."
        )
    except Exception as e:
        print(
            f"Warning: Error loading config file '{args.config_path}': {e}. Proceeding without full language list."
        )

    # 1. Extract Data
    print(f"\nExtracting detailed data from: {args.results_dir}")
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        return

    print(f"\nExtracted {len(df_data)} data points (rows).")
    print("Columns found:", df_data.columns.tolist())
    print("Metric types found:", df_data["metric_type"].unique())

    # 2. Prepare for Plotting
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Determine which plots to generate
    plot_types_to_generate = []
    if args.plot_type == "all":
        # --- MODIFICATION START: Added new types to 'all' ---
        plot_types_to_generate = [
            "boxplot",
            "heatmap",
            "dataset_boxplot",
            "algo_vs_model_f1",
            "algo_vs_model_success"
            ]
        # --- MODIFICATION END ---
    elif args.plot_type in ["boxplot", "heatmap", "dataset_boxplot", "algo_vs_model_f1", "algo_vs_model_success"]:
        plot_types_to_generate = [args.plot_type]
    else:
        print(f"Error: Invalid plot type '{args.plot_type}' specified.")
        sys.exit(1)

    # 4. Generate Plots
    for plot_type in plot_types_to_generate:
        print(f"\n--- Generating {plot_type.replace('_', ' ').title()} ---")

        if plot_type == "boxplot":
            # Call the F1 Boxplot Generator
            generate_f1_boxplot(
                df_data=df_data,
                group_by=args.group_by,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                all_languages_list=all_languages_list,
            )

        elif plot_type == "heatmap":
            # --- Call the F1 Score Heatmap Generators ---
            print("\nGenerating Detailed F1 Score Heatmaps (Lang/Chunk/Model)")
            # Filter data specifically for F1 scores for these heatmaps
            df_f1_heatmap = df_data[df_data["metric_type"] == "f1_score"].copy()
            if df_f1_heatmap.empty:
                print("Warning: No F1 score data found. Skipping detailed F1 heatmaps.")
            else:
                # Call Language vs Model heatmap (includes zeroshot)
                generate_language_vs_model_heatmap(
                    df_f1_heatmap=df_f1_heatmap,
                    output_dir=args.output_dir,
                    output_filename_prefix=args.output_filename_prefix,
                    all_languages_list=all_languages_list,  # Pass language list
                )

                # Filter out 'zeroshot' for heatmaps requiring chunk/overlap
                df_f1_heatmap_rag = df_f1_heatmap[
                    df_f1_heatmap["retrieval_algorithm"] != "zeroshot"
                ].copy()

                if df_f1_heatmap_rag.empty:
                    print(
                        "Warning: No F1 score data found for non-zeroshot algorithms. Skipping Chunk/Overlap related F1 heatmaps."
                    )
                # else:
                    # Call Chunk vs Overlap heatmap (excludes zeroshot)
                    # generate_chunk_vs_overlap_heatmap(
                    #     df_f1_heatmap=df_f1_heatmap_rag, # Use filtered data
                    #     output_dir=args.output_dir,
                    #     output_filename_prefix=args.output_filename_prefix,
                    # )
                    # Call Model vs Chunk/Overlap heatmap (excludes zeroshot)
                    # generate_model_vs_chunk_overlap_heatmap(
                    #     df_f1_heatmap=df_f1_heatmap_rag, # Use filtered data
                    #     output_dir=args.output_dir,
                    #     output_filename_prefix=args.output_filename_prefix,
                    # )

            # --- Call the Detailed Dataset Success Rate Heatmap Generators ---
            # This function handles its own filtering internally if needed
            # print("\nGenerating Detailed Dataset Success Heatmaps (Model vs Chunk/Overlap)")
            # generate_dataset_success_heatmaps(
            #     df_data=df_data,  # Pass the full dataframe
            #     output_dir=args.output_dir,
            #     output_filename_prefix=args.output_filename_prefix,
            # )

        elif plot_type == "dataset_boxplot":
            # Call the Dataset Success Boxplot Generator
            generate_dataset_success_boxplot(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                lang=DATASET_BOXPLOT_LANG,
                chunk=DATASET_BOXPLOT_CHUNK,
                overlap=DATASET_BOXPLOT_OVERLAP,
            )

        # --- NEW: Add calls for the new summary heatmaps ---
        elif plot_type == "algo_vs_model_f1":
            generate_algo_vs_model_f1_heatmap(
                df_data=df_data, # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )

        elif plot_type == "algo_vs_model_success":
            generate_algo_vs_model_dataset_success_heatmap(
                df_data=df_data, # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )
        # --- END NEW ---


    print("\n--- Visualization Generation Finished ---")


if __name__ == "__main__":
    main()
