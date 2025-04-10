# visualization/plot_scripts/main_visualization.py
import os
import sys
import argparse
import pandas as pd
from collections import Counter  # Import Counter
import re # Import re for filename sanitization

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
    # Import all three heatmap functions
    from visualization.plot_scripts.heatmaps import (
        create_f1_heatmap,
        create_chunk_overlap_heatmap,
        create_model_vs_chunk_overlap_heatmap
    )
except ImportError as e:
    print("Error importing required modules.")
    print(f"Please ensure the script is run from a location where Python can find:")
    print(f"  - '{project_root_dir}/utils/config_loader.py'")  # Added ConfigLoader path
    print(f"  - '{project_root_dir}/visualization/visualization_data_extractor.py'")
    print(f"  - '{visualization_dir}/plot_scripts/box_plots.py'")
    # Update path in error message
    print(f"  - '{visualization_dir}/plot_scripts/heatmaps.py'")
    print(f"Current sys.path: {sys.path}")
    print(f"Original Error: {e}")
    sys.exit(1)
def sanitize_filename(name: str) -> str:
    """Removes or replaces characters problematic for filenames."""
    # Remove or replace characters like ':', '/', '\', etc.
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    # Replace sequences of whitespace or underscores with a single underscore
    name = re.sub(r'[\s_]+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name if name else "invalid_name"



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
            print("\n--- Generating Heatmaps ---")

            # --- Heatmap Type 1: Language vs Model (per Algo/Chunk/Overlap) ---
            print("\nGenerating Heatmaps: Language vs Model (per Algo/Chunk/Overlap)")
            grouping_columns_lang_model = ['retrieval_algorithm', 'chunk_size', 'overlap_size']
            required_cols_lang_model = grouping_columns_lang_model + ['language', 'question_model', 'f1_score']

            if not all(col in df_data.columns for col in required_cols_lang_model):
                print(f"Warning: DataFrame is missing one or more required columns for Language vs Model heatmap grouping: {required_cols_lang_model}. Skipping this heatmap type.")
                # Continue to the next heatmap type if possible
            else:
                try:
                    unique_combinations_lang_model = df_data[grouping_columns_lang_model].drop_duplicates().to_dict('records')
                except KeyError as e:
                    print(f"Error: Could not find grouping columns {grouping_columns_lang_model} for Language vs Model heatmaps. Error: {e}. Skipping this heatmap type.")
                    unique_combinations_lang_model = [] # Prevent further processing

                if not unique_combinations_lang_model:
                    print("No unique combinations of (retrieval_algorithm, chunk_size, overlap_size) found for Language vs Model heatmaps.")
                else:
                    print(f"Found {len(unique_combinations_lang_model)} unique parameter combinations for Language vs Model heatmaps.")

                    for i, combo in enumerate(unique_combinations_lang_model):
                        algo = combo['retrieval_algorithm']
                        cs = combo['chunk_size']
                        ov = combo['overlap_size']

                        print(f"\n[{i+1}/{len(unique_combinations_lang_model)}] Generating Language vs Model Heatmap for: Algo={algo}, Chunk={cs}, Overlap={ov}")

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
                        combo_str = f"algo_{sanitize_filename(str(algo))}_cs_{cs}_os_{ov}"
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
                            current_params=combo
                        )

            # --- Heatmap Type 2: Chunk Size vs Overlap Size (per Lang/Model/Algo) ---
            print("\nGenerating Heatmaps: Chunk Size vs Overlap Size (per Lang/Model/Algo)")
            grouping_columns_chunk_overlap = ['language', 'question_model', 'retrieval_algorithm']
            required_cols_chunk_overlap = grouping_columns_chunk_overlap + ['chunk_size', 'overlap_size', 'f1_score']

            if not all(col in df_data.columns for col in required_cols_chunk_overlap):
                 print(f"Warning: DataFrame is missing one or more required columns for Chunk/Overlap heatmap grouping: {required_cols_chunk_overlap}. Skipping this heatmap type.")
                 # Continue to the next heatmap type if possible
            else:
                try:
                    unique_fixed_params_chunk = df_data[grouping_columns_chunk_overlap].drop_duplicates().to_dict('records')
                except KeyError as e:
                    print(f"Error: Could not find grouping columns {grouping_columns_chunk_overlap} for Chunk/Overlap heatmaps. Error: {e}. Skipping this heatmap type.")
                    unique_fixed_params_chunk = [] # Prevent further processing

                if not unique_fixed_params_chunk:
                    print("No unique combinations of (language, question_model, retrieval_algorithm) found for Chunk/Overlap heatmaps.")
                else:
                    print(f"Found {len(unique_fixed_params_chunk)} unique fixed parameter combinations for Chunk/Overlap heatmaps.")
                    for i, fixed_params in enumerate(unique_fixed_params_chunk):
                        lang = fixed_params['language']
                        model = fixed_params['question_model']
                        algo = fixed_params['retrieval_algorithm']
                        print(f"\n[{i+1}/{len(unique_fixed_params_chunk)}] Generating Chunk/Overlap Heatmap for: Lang={lang}, Model={model}, Algo={algo}")

                        filtered_df_chunk_combo = df_data[
                            (df_data['language'] == lang) &
                            (df_data['question_model'] == model) &
                            (df_data['retrieval_algorithm'] == algo)
                        ].copy()

                        # Check if there's enough variation to plot (at least 2x2 grid potential)
                        if filtered_df_chunk_combo.empty or \
                           filtered_df_chunk_combo['chunk_size'].nunique() < 2 or \
                           filtered_df_chunk_combo['overlap_size'].nunique() < 2:
                            print("  Skipping: Not enough data or variation in chunk/overlap sizes (need at least 2 unique chunk and 2 unique overlap values) for this combination.")
                            continue

                        print(f"  Data points for this combination: {len(filtered_df_chunk_combo)}")

                        sanitized_model = sanitize_filename(str(model))
                        sanitized_algo = sanitize_filename(str(algo))
                        sanitized_lang = sanitize_filename(str(lang))
                        fixed_param_str = f"lang_{sanitized_lang}_model_{sanitized_model}_algo_{sanitized_algo}"
                        output_filename = f"{args.output_filename_prefix}f1_heatmap_chunk_vs_overlap_{fixed_param_str}.png"
                        output_filepath = os.path.join(args.output_dir, output_filename)

                        create_chunk_overlap_heatmap(
                            data=filtered_df_chunk_combo,
                            output_path=output_filepath,
                            fixed_params=fixed_params,
                            values_col='f1_score',
                            index_col='chunk_size',
                            columns_col='overlap_size'
                        )

            # --- Heatmap Type 3: Model vs Chunk/Overlap (English Only, per Algo) ---
            print("\nGenerating Heatmaps: Model vs Chunk/Overlap (English Only, per Algo)")
            required_cols_model_chunk = ['language', 'retrieval_algorithm', 'question_model', 'chunk_size', 'overlap_size', 'f1_score']

            if not all(col in df_data.columns for col in required_cols_model_chunk):
                 print(f"Warning: DataFrame is missing one or more required columns for Model vs Chunk/Overlap heatmap: {required_cols_model_chunk}. Skipping this heatmap type.")
                 # End of heatmap generation for this loop iteration
            else:
                # Filter for English language first
                df_english = df_data[df_data['language'] == 'english'].copy()

                if df_english.empty:
                    print("No data found for language 'english'. Skipping Model vs Chunk/Overlap heatmaps.")
                else:
                    try:
                        unique_algos_english = df_english['retrieval_algorithm'].unique().tolist()
                    except KeyError:
                        print("Error: 'retrieval_algorithm' column not found in English data. Skipping Model vs Chunk/Overlap heatmaps.")
                        unique_algos_english = []

                    if not unique_algos_english:
                        print("No unique retrieval algorithms found within English data.")
                    else:
                        print(f"Found {len(unique_algos_english)} algorithms for English Model vs Chunk/Overlap heatmaps: {unique_algos_english}")

                        for i, algo in enumerate(unique_algos_english):
                            print(f"\n[{i+1}/{len(unique_algos_english)}] Generating Model vs Chunk/Overlap Heatmap for: Lang=english, Algo={algo}")

                            filtered_df_model_chunk_combo = df_english[
                                df_english['retrieval_algorithm'] == algo
                            ].copy()

                            # Check if data exists and has enough variation (at least one model and one chunk/overlap combo)
                            if filtered_df_model_chunk_combo.empty or \
                               filtered_df_model_chunk_combo['chunk_size'].nunique() < 1 or \
                               filtered_df_model_chunk_combo['overlap_size'].nunique() < 1 or \
                               filtered_df_model_chunk_combo['question_model'].nunique() < 1:
                                print("  Skipping: Not enough data or variation (need at least 1 model, 1 chunk size, 1 overlap size) for this combination.")
                                continue

                            print(f"  Data points for this combination: {len(filtered_df_model_chunk_combo)}")

                            fixed_params_model_chunk = {'language': 'english', 'retrieval_algorithm': algo}

                            sanitized_algo = sanitize_filename(str(algo))
                            fixed_param_str = f"lang_english_algo_{sanitized_algo}"
                            output_filename = f"{args.output_filename_prefix}f1_heatmap_model_vs_chunk_overlap_{fixed_param_str}.png"
                            output_filepath = os.path.join(args.output_dir, output_filename)

                            create_model_vs_chunk_overlap_heatmap(
                                data=filtered_df_model_chunk_combo,
                                output_path=output_filepath,
                                fixed_params=fixed_params_model_chunk,
                                values_col='f1_score',
                                index_col_chunk='chunk_size',
                                index_col_overlap='overlap_size',
                                columns_col='question_model'
                            )

    print("\n--- Visualization Generation Finished ---")


if __name__ == "__main__":
    main()
