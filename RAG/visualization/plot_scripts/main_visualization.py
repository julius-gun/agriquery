# visualization/plot_scripts/main_visualization.py

# python visualization\plot_scripts\main_visualization.py --plot-type=heatmap
# python visualization\plot_scripts\main_visualization.py --plot-type=all

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
    from plot_utils import add_project_paths, sanitize_filename, get_model_colors

    PROJECT_ROOT = add_project_paths()  # Ensure paths are set and get the root
except ImportError as e:
    print(
        "Failed to import 'add_project_paths' or 'sanitize_filename' from 'plot_utils.py'."
    )
    print(
        f"Make sure 'plot_utils.py' exists in the same directory as this script ({os.path.dirname(__file__)}) and is correctly structured."
    )
    print(f"Original error: {e}")
    # Attempt to provide more context on where it's looking
    print(f"Current sys.path: {sys.path}")
    # Try calculating expected path for plot_utils
    expected_plot_utils_path = os.path.join(os.path.dirname(__file__), "plot_utils.py")
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

    # Import the heatmap generator functions
    from visualization.plot_scripts.heatmap_generators import (
        generate_language_vs_model_heatmap,
        generate_chunk_vs_overlap_heatmap,
        generate_model_vs_chunk_overlap_heatmap,
        generate_dataset_success_heatmaps,  # The detailed one: Model vs Chunk/Overlap for Dataset Success
        generate_algo_vs_model_f1_heatmap,  # English only, mean F1
        generate_algo_vs_model_dataset_success_heatmap,  # Multi-lang, algo-sorted, per-dataset success
        generate_multilang_f1_score_report_heatmap,
        generate_multilang_accuracy_report_heatmap,  # NEW
        generate_multilang_precision_report_heatmap,  # NEW
        generate_multilang_recall_report_heatmap,  # NEW
        generate_multilang_specificity_report_heatmap,  # NEW
    )
    # --- MODIFICATION START: Import line chart generator ---
    from visualization.plot_scripts.linechart_generators import generate_zeroshot_performance_linecharts
    # --- MODIFICATION END ---

except ImportError as e:
    print("Error importing required modules after attempting path setup.")
    print(
        "This might indicate an issue with the project structure or missing files within 'utils' or 'visualization'."
    )
    print(
        f"Project root added to sys.path: {PROJECT_ROOT if 'PROJECT_ROOT' in locals() else 'Unknown'}"
    )
    print(f"Current sys.path: {sys.path}")
    print(f"Original Error: {e}")
    sys.exit(1)
# ... (potentially barchart_generators if you have them)
# from visualization.plot_scripts.barchart_generators import generate_some_barcharts

# Define constants for plot types to generate
# PLOT_TYPE_BARCHARTS = "barcharts" # If you add barcharts


# sanitize_filename function is now imported from plot_utils

# --- Global Plotting Configuration ---
# Define the desired sorting order for models in specific reports
REPORT_MODEL_SORT_ORDER = [
    "gemini-2.5-flash-preview-04-17",
    "qwen2.5_7B-128k",
    "qwen3_8B-128k",
    "phi3_14B_q4_medium-128k",
    "llama3.1_8B-128k",
    "deepseek-r1_8B-128k",
    "llama3.2_3B-128k",
    "deepseek-r1_1.5B-128k",
    "llama3.2_1B-128k"
]


def main():
    # Get project root and visualization dir using the established PROJECT_ROOT
    visualization_dir = os.path.join(PROJECT_ROOT, "visualization")  # Derived from root
    project_root_dir = PROJECT_ROOT  # Use the established root

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
            "boxplot",
            "heatmap",  # Original set of detailed heatmaps (lang_vs_model, chunk_vs_overlap etc.)
            "zeroshot_linecharts", # Added new plot type
            "dataset_boxplot",
            "algo_vs_model_f1",  # English only, mean F1
            "algo_vs_model_success",  # Multi-lang, algo-sorted, per-dataset success
            "multilang_f1_report",
            "multilang_accuracy_report",
            "multilang_precision_report",
            "multilang_recall_report",
            "multilang_specificity_report",
            "all",
        ],
        help="Type of plot(s) to generate.",
    )
    parser.add_argument(
        "--group-by",
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
    print(f"Using model sort order for reports: {REPORT_MODEL_SORT_ORDER}")


    # 0. Load Config to get languages
    all_languages_list = None
    config_loader = None # Initialize config_loader
    try:
        if os.path.exists(args.config_path):
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
        else:
            print(
                f"Warning: Config file not found at '{args.config_path}'. Cannot determine full language list for consistent plot elements or model colors from config."
            )

    except FileNotFoundError: # Should be caught by os.path.exists, but good to have
        print(
            f"Warning: Config file not found at '{args.config_path}'. Cannot determine full language list for consistent plot elements or model colors from config."
        )
    except Exception as e:
        print(
            f"Warning: Error loading config file '{args.config_path}': {e}. Proceeding without full language list or model colors from config."
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

    # --- MODIFICATION START: Generate master model palette ---
    master_model_palette = None
    if df_data is not None and not df_data.empty and "question_model" in df_data.columns:
        all_models_in_data = df_data["question_model"].unique().tolist()
        current_config = config_loader.config if config_loader and hasattr(config_loader, 'config') else None
        master_model_palette = get_model_colors(all_models_in_data, current_config)
        
        if master_model_palette:
            print(f"Generated master model color palette for {len(master_model_palette)} models.")
        else:
            print("Warning: Could not generate master model color palette. Seaborn defaults will be used by plots requiring it.")
    else:
        print("Warning: Cannot generate master model palette due to missing data or 'question_model' column.")
    # --- MODIFICATION END ---

    # 3. Determine which plots to generate
    plot_types_to_generate = []
    if args.plot_type == "all":
        # --- MODIFICATION START: Added new types to 'all' ---
        plot_types_to_generate = [
            "boxplot",
            "heatmap",
            "zeroshot_linecharts", # Added new plot type
            "dataset_boxplot",
            "algo_vs_model_f1",  # English-only, mean F1
            "algo_vs_model_success",  # Multi-lang, per-dataset success
            "multilang_f1_report",
            "multilang_accuracy_report",
            "multilang_precision_report",
            "multilang_recall_report",
            "multilang_specificity_report",
        ]
    elif args.plot_type in [  # Add new types to this list
        "boxplot",
        "heatmap",
        "zeroshot_linecharts", # Added new plot type
        "dataset_boxplot",
        "algo_vs_model_f1",
        "algo_vs_model_success",
        "multilang_f1_report",
        "multilang_accuracy_report",
        "multilang_precision_report",
        "multilang_recall_report",
        "multilang_specificity_report",
    ]:
        plot_types_to_generate = [args.plot_type]
    else:
        print(f"Error: Invalid plot type '{args.plot_type}' specified.")
        sys.exit(1)

    # 4. Generate Plots
    for plot_type in plot_types_to_generate:
        print(f"\n--- Generating {plot_type.replace('_', ' ').title()} ---")

        if plot_type == "boxplot":
            ...
            # Call the F1 Boxplot Generator for distribution by question model and language
            # generate_f1_boxplot(
            #     df_data=df_data,
            #     group_by=args.group_by,
            #     output_dir=args.output_dir,
            #     output_filename_prefix=args.output_filename_prefix,
            #     all_languages_list=all_languages_list,
            # )
            pass # Placeholder for brevity, no change requested here yet

        elif plot_type == "heatmap": # This is for the set of original detailed heatmaps
            print("\nGenerating Detailed F1 Score & Dataset Success Heatmaps (Original Set)")
            # Filter data specifically for F1 scores for these heatmaps
            df_f1_heatmap = df_data[df_data["metric_type"] == "f1_score"].copy()
            if df_f1_heatmap.empty:
                print("Warning: No F1 score data found. Skipping detailed F1 heatmaps for 'heatmap' type.")
            else:
                # Call Language vs Model heatmap (includes zeroshot)
                # generate_language_vs_model_heatmap(
                #     df_f1_heatmap=df_f1_heatmap,
                #     output_dir=args.output_dir,
                #     output_filename_prefix=args.output_filename_prefix,
                #     all_languages_list=all_languages_list,  # Pass language list
                #     # If this heatmap needs model sorting, pass REPORT_MODEL_SORT_ORDER
                # )
                pass
                # Filter out 'zeroshot' for heatmaps requiring chunk/overlap
                df_f1_heatmap_rag = df_f1_heatmap[
                    df_f1_heatmap["retrieval_algorithm"] != "zeroshot"
                ].copy()

                if not df_f1_heatmap_rag.empty:
                    # generate_chunk_vs_overlap_heatmap(
                    #     df_f1_heatmap=df_f1_heatmap_rag,
                    #     output_dir=args.output_dir,
                    #     output_filename_prefix=args.output_filename_prefix,
                    # )
                    # generate_model_vs_chunk_overlap_heatmap(
                    #     df_f1_heatmap=df_f1_heatmap_rag,
                    #     output_dir=args.output_dir,
                    #     output_filename_prefix=args.output_filename_prefix,
                    # )
                    pass # Placeholder for brevity
                else:
                    print(
                        "Warning: No F1 score data found for non-zeroshot algorithms. Skipping Chunk/Overlap and Model vs Chunk/Overlap F1 heatmaps."
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
            pass # Placeholder for brevity


        elif plot_type == "dataset_boxplot":
            # Call the Dataset Success Boxplot Generator
            # generate_dataset_success_boxplot(
            #     df_data=df_data,
            #     output_dir=args.output_dir,
            #     output_filename_prefix=args.output_filename_prefix,
            #     lang=DATASET_BOXPLOT_LANG,
            #     chunk=DATASET_BOXPLOT_CHUNK,
            #     overlap=DATASET_BOXPLOT_OVERLAP,
            # )
            pass

        # --- NEW: Add calls for the new summary heatmaps ---
        elif plot_type == "algo_vs_model_f1":
            # generate_algo_vs_model_f1_heatmap(
            #     df_data=df_data,  # Pass the full dataframe
            #     output_dir=args.output_dir,
            #     output_filename_prefix=args.output_filename_prefix,
            # )
            pass

        elif plot_type == "algo_vs_model_success":
            generate_algo_vs_model_dataset_success_heatmap(
                df_data=df_data,  # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                model_sort_order=REPORT_MODEL_SORT_ORDER # Pass the sort order
            )

        elif plot_type == "multilang_f1_report":  # Add elif for the new plot
            generate_multilang_f1_score_report_heatmap(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                model_sort_order=REPORT_MODEL_SORT_ORDER # Pass the sort order
            )
        
        elif plot_type == "multilang_accuracy_report":
            generate_multilang_accuracy_report_heatmap(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                model_sort_order=REPORT_MODEL_SORT_ORDER # Pass the sort order
            )

        elif plot_type == "multilang_precision_report":
            generate_multilang_precision_report_heatmap(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                model_sort_order=REPORT_MODEL_SORT_ORDER # Pass the sort order
            )

        elif plot_type == "multilang_recall_report":
            generate_multilang_recall_report_heatmap(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                model_sort_order=REPORT_MODEL_SORT_ORDER # Pass the sort order
            )

        elif plot_type == "multilang_specificity_report":
            generate_multilang_specificity_report_heatmap(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                model_sort_order=REPORT_MODEL_SORT_ORDER # Pass the sort order
            )
        elif plot_type == "zeroshot_linecharts":
            if df_data is not None and not df_data.empty:
                generate_zeroshot_performance_linecharts(
                    df_data=df_data,
                    output_dir=args.output_dir,
                    output_filename_prefix=args.output_filename_prefix,
                    model_sort_order=REPORT_MODEL_SORT_ORDER,
                    model_palette=master_model_palette, # Use the generated master palette
                    languages_to_plot=all_languages_list # Pass the list of languages from config
                    # figsize is left to default in the generator function
                )
            else:
                print("Skipping zero-shot line charts as no data is available.")

    print("\n--- Visualization Generation Finished ---")


if __name__ == "__main__":
    main()
