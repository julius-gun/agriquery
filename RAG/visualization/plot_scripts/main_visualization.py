# visualization/plot_scripts/main_visualization.py
import os
import sys
import argparse
import pandas as pd
from collections import Counter  # Import Counter

# --- Adjust Python Path ---
# Add the parent directory ('visualization') and the project root ('RAG') to the Python path
# This allows importing modules from sibling directories (like visualization_data_extractor)
# and the plotting script itself.
current_dir = os.path.dirname(os.path.abspath(__file__))
visualization_dir = os.path.dirname(current_dir)
project_root_dir = os.path.dirname(
    visualization_dir
)  # Assumes RAG is the parent of visualization

# Add project root to path FIRST to prioritize project modules
sys.path.insert(0, project_root_dir)
# Add visualization directory to path
sys.path.insert(1, visualization_dir)

# --- Imports ---
# Now we can import using the adjusted path
try:
    from utils.config_loader import ConfigLoader  # Import ConfigLoader
    from visualization.visualization_data_extractor import extract_visualization_data
    from visualization.plot_scripts.box_plots import create_f1_boxplot
    from visualization.plot_scripts.heatmaps import create_f1_heatmap
except ImportError as e:
    print("Error importing required modules.")
    print(f"Please ensure the script is run from a location where Python can find:")
    print(f"  - '{project_root_dir}/utils/config_loader.py'")  # Added ConfigLoader path
    print(f"  - '{project_root_dir}/visualization/visualization_data_extractor.py'")
    print(f"  - '{visualization_dir}/plot_scripts/box_plots.py'")
    print(f"  - '{visualization_dir}/plot_scripts/heatmaps.py'")
    print(f"Current sys.path: {sys.path}")
    print(f"Original Error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for RAG test F1 scores."
    )
    parser.add_argument(
        "--config-path",  # Add argument for config path
        type=str,
        default=os.path.join(project_root_dir, "config.json"),
        help="Path to the configuration JSON file (to get language list).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(project_root_dir, "results"),
        help="Directory containing the JSON result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(visualization_dir, "plots"),
        help="Directory where the generated plot image will be saved.",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",  # Default changed to 'all'
        choices=["boxplot", "heatmap", "all"],
        help="Type of plot(s) to generate. 'all' generates default configurations of each type.",
    )

    # --- Boxplot Specific Arguments ---
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
        help="Parameter to group the box plots by (used if plot-type is 'boxplot' or 'all').",
    )

    # --- Heatmap Specific Arguments (Filters) ---
    # Set default values for filters
    parser.add_argument(
        "--filter-algo",
        type=str,
        # default='embedding', # Default value removed - let heatmap function handle None
        help="Filter heatmap data by retrieval_algorithm (e.g., 'embedding'). If omitted, no filter applied for this param.",
    )
    parser.add_argument(
        "--filter-chunk",
        type=int,
        # default=2000, # Default value removed
        help="Filter heatmap data by chunk_size (e.g., 2000). If omitted, no filter applied for this param.",
    )
    parser.add_argument(
        "--filter-overlap",
        type=int,
        # default=100, # Default value removed
        help="Filter heatmap data by overlap_size (e.g., 100). If omitted, no filter applied for this param.",
    )
    parser.add_argument(
        "--filter-topk",
        type=int,
        # default=3, # Default value removed
        help="Filter heatmap data by num_retrieved_docs (top_k) (e.g., 3). If omitted, no filter applied for this param.",
    )
    # Note: Heatmap index/columns are fixed to language/model for now, but could be args too

    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="",
        help="Optional prefix for all generated plot filenames (e.g., 'run1_').",
    )

    args = parser.parse_args()

    print("--- Starting Visualization Generation ---")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Requested plot type(s): {args.plot_type}")
    if args.output_filename_prefix:
        print(f"Filename prefix: {args.output_filename_prefix}")

    # 0. Load Config to get languages
    try:
        config_loader = ConfigLoader(args.config_path)
        language_configs = config_loader.config.get("language_configs", [])
        all_languages_list = [
            lc.get("language") for lc in language_configs if lc.get("language")
        ]
        if not all_languages_list:
            print(
                f"Warning: No languages found in 'language_configs' in {args.config_path}. Heatmap might not show all expected languages."
            )
            all_languages_list = None  # Set to None if empty
        else:
            print(f"Found languages in config: {all_languages_list}")
    except FileNotFoundError:
        print(
            f"Error: Config file not found at '{args.config_path}'. Cannot determine full language list."
        )
        all_languages_list = None  # Proceed without the full list
    except Exception as e:
        print(f"Error loading config file '{args.config_path}': {e}")
        all_languages_list = None  # Proceed without the full list

    # 1. Extract Data
    df_data = extract_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted.")
        return

    # 2. Prepare for Plotting
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Determine which plots to generate
    plot_types_to_generate = []
    if args.plot_type == "all":
        plot_types_to_generate = ["boxplot", "heatmap"]
    elif args.plot_type in ["boxplot", "heatmap"]:
        plot_types_to_generate = [args.plot_type]
    else:
        print(
            f"Error: Invalid plot type '{args.plot_type}' specified."
        )  # Should be caught by choices, but good practice
        sys.exit(1)

    # 4. Generate Plots
    for plot_type in plot_types_to_generate:
        print(f"\n--- Generating {plot_type.capitalize()} ---")

        if plot_type == "boxplot":
            print(f"Grouping by: {args.group_by}")

            # Generate filename
            output_filename = (
                f"{args.output_filename_prefix}f1_boxplot_by_{args.group_by}.png"
            )
            output_filepath = os.path.join(args.output_dir, output_filename)

            create_f1_boxplot(
                data=df_data,
                group_by_column=args.group_by,
                output_path=output_filepath,
            )

        elif plot_type == "heatmap":
            print("Target: Language vs. Question Model")

            # --- Construct Filters (only include if argument was provided) ---
            filter_params = {}
            if args.filter_algo is not None:
                filter_params["retrieval_algorithm"] = args.filter_algo
            if args.filter_chunk is not None:
                filter_params["chunk_size"] = args.filter_chunk
            if args.filter_overlap is not None:
                filter_params["overlap_size"] = args.filter_overlap
            if args.filter_topk is not None:
                filter_params["num_retrieved_docs"] = args.filter_topk

            if filter_params:
                print(f"Applying filters: {filter_params}")
            else:
                print("No filters applied for heatmap.")

            # --- Generate Filename ---
            if filter_params:
                filter_str = "_".join(
                    f"{k.replace('_', '').replace('retrievalalgorithm', 'algo').replace('numretrieveddocs', 'k')}{v}"
                    for k, v in filter_params.items()
                )
                output_filename = f"{args.output_filename_prefix}f1_heatmap_lang_vs_model_{filter_str}.png"
            else:
                output_filename = f"{args.output_filename_prefix}f1_heatmap_lang_vs_model_all_data.png"  # Indicate no filter

            output_filepath = os.path.join(args.output_dir, output_filename)

            # --- Create Heatmap ---
            # Pass the original df_data, filtering happens inside create_f1_heatmap
            # Pass the list of all languages obtained from the config
            create_f1_heatmap(
                data=df_data,  # Pass the full extracted data
                output_path=output_filepath,
                filter_params=filter_params
                if filter_params
                else None,  # Pass filters or None
                all_indices=all_languages_list,  # Pass the full language list here
                index_col="language",  # Fixed for this specific heatmap request
                columns_col="question_model",  # Fixed for this specific heatmap request
                values_col="f1_score",
            )

    print("\n--- Visualization Generation Finished ---")


if __name__ == "__main__":
    main()
