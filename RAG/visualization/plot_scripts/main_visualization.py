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
        help="Directory where the generated plot images will be saved.",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["boxplot", "heatmap", "all"],
        help="Type of plot(s) to generate. 'all' generates default boxplot and iterates through heatmap combinations.",
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

    # --- REMOVED Heatmap Filter Arguments ---
    # Filtering is now done by iterating through unique combinations found in the data.

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

    # 0. Load Config to get languages (needed for consistent heatmap rows)
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
            all_languages_list = None # Heatmap function will use only languages present in data
        else:
            print(f"Found languages in config for consistent heatmap rows: {all_languages_list}")
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at '{args.config_path}'. Cannot determine full language list for consistent heatmap rows."
        )
        all_languages_list = None # Proceed without the full list
    except Exception as e:
        print(f"Warning: Error loading config file '{args.config_path}': {e}. Proceeding without full language list.")
        all_languages_list = None # Proceed without the full list

    # 1. Extract Data
    print(f"\nExtracting data from: {args.results_dir}")
    df_data = extract_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        return

    # Display basic info about extracted data
    print(f"\nExtracted {len(df_data)} data points.")
    print("Columns found:", df_data.columns.tolist())
    # print("Data head:\n", df_data.head()) # Uncomment for debugging

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
        # This case should be caught by argparse choices, but good practice
        print(f"Error: Invalid plot type '{args.plot_type}' specified.")
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
            # Identify unique combinations of parameters to iterate over
            grouping_columns = ['retrieval_algorithm', 'chunk_size', 'overlap_size']
            if not all(col in df_data.columns for col in grouping_columns):
                print(f"Error: DataFrame is missing one or more required columns for heatmap grouping: {grouping_columns}")
                print(f"Available columns: {df_data.columns.tolist()}")
                continue # Skip heatmap generation if columns are missing

            try:
                # drop_duplicates returns a DataFrame, convert to list of dicts
                unique_combinations = df_data[grouping_columns].drop_duplicates().to_dict('records')
            except KeyError as e:
                 print(f"Error: Could not find grouping columns {grouping_columns} in DataFrame. Error: {e}")
                 continue # Skip heatmap generation

            if not unique_combinations:
                print("No unique combinations of (retrieval_algorithm, chunk_size, overlap_size) found in the data.")
            else:
                print(f"Found {len(unique_combinations)} unique parameter combinations for heatmaps.")

                for i, combo in enumerate(unique_combinations):
                    algo = combo['retrieval_algorithm']
                    cs = combo['chunk_size']
                    ov = combo['overlap_size'] # Changed variable name for clarity

                    print(f"\n[{i+1}/{len(unique_combinations)}] Generating Heatmap for: Algo={algo}, Chunk={cs}, Overlap={ov}")

                    # Filter data for this specific combination
                    # Use boolean indexing for clarity and efficiency
                    filtered_df_combo = df_data[
                        (df_data['retrieval_algorithm'] == algo) &
                        (df_data['chunk_size'] == cs) &
                        (df_data['overlap_size'] == ov)
                    ].copy() # Use .copy() to avoid SettingWithCopyWarning if modifying later

                    if filtered_df_combo.empty:
                        print("  No data found for this specific combination. Skipping heatmap.")
                        continue

                    print(f"  Data points for this combination: {len(filtered_df_combo)}")

                    # Generate filename including the parameters
                    combo_str = f"algo_{algo}_cs_{cs}_os_{ov}"
                    output_filename = f"{args.output_filename_prefix}f1_heatmap_lang_vs_model_{combo_str}.png"
                    output_filepath = os.path.join(args.output_dir, output_filename)

                    # Call heatmap function with filtered data and combo details
                    create_f1_heatmap(
                        data=filtered_df_combo, # Pass the filtered data
                        output_path=output_filepath,
                        all_indices=all_languages_list, # Pass the full language list for consistent rows
                        index_col="language",
                        columns_col="question_model",
                        values_col="f1_score",
                        current_params=combo # Pass the dictionary of current parameters for the title
                    )

    print("\n--- Visualization Generation Finished ---")


if __name__ == "__main__":
    main()
