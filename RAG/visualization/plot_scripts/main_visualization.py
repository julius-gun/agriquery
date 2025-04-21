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
    # Use the renamed data extractor
    from visualization.visualization_data_extractor import extract_detailed_visualization_data
    # Import both boxplot functions
    from visualization.plot_scripts.box_plots import create_f1_boxplot, create_dataset_success_boxplot
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
        description="Generate visualizations for RAG test results."
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
        # Add the new plot type choice
        choices=["boxplot", "heatmap", "dataset_boxplot", "all"],
         help="Type of plot(s) to generate. 'dataset_boxplot' generates dataset success plots for specific params. 'all' generates default boxplot, heatmaps, and dataset_boxplot.",
    )

    # --- Boxplot Specific Arguments (for the original F1 boxplot) ---
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
        help="Parameter to group the F1 box plots by (used if plot-type is 'boxplot' or 'all').",
    )
    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="",
        help="Optional prefix for all generated plot filenames (e.g., 'run1_').",
    )


    # --- Constants for dataset_boxplot ---
    DATASET_BOXPLOT_LANG = 'english'
    DATASET_BOXPLOT_CHUNK = 200
    DATASET_BOXPLOT_OVERLAP = 100

    args = parser.parse_args()

    print("--- Starting Visualization Generation ---")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Requested plot type(s): {args.plot_type}")
    if args.plot_type == 'dataset_boxplot' or args.plot_type == 'all':
        print(f"Dataset Boxplot Params: Lang={DATASET_BOXPLOT_LANG}, Chunk={DATASET_BOXPLOT_CHUNK}, Overlap={DATASET_BOXPLOT_OVERLAP}")
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
            print(f"Found languages in config for consistent plot elements: {all_languages_list}")
        else:
            print(
                f"Warning: No languages found in 'language_configs' in {args.config_path}. Boxplot legend order may vary."
            )
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at '{args.config_path}'. Cannot determine full language list for consistent plot elements."
        )
    except Exception as e:
        print(f"Warning: Error loading config file '{args.config_path}': {e}. Proceeding without full language list.")

    # 1. Extract Data using the new detailed extractor
    print(f"\nExtracting detailed data from: {args.results_dir}")
    # Call the renamed extractor
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        return

    # Display basic info about extracted data
    print(f"\nExtracted {len(df_data)} data points (rows).")
    print("Columns found:", df_data.columns.tolist())
    print("Metric types found:", df_data['metric_type'].unique())
    # print("Data head:\n", df_data.head()) # Uncomment for debugging

    # 2. Prepare for Plotting
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Determine which plots to generate
    plot_types_to_generate = []
    if args.plot_type == "all":
        # Include the new type in 'all'
        plot_types_to_generate = ["boxplot", "heatmap", "dataset_boxplot"]
    elif args.plot_type in ["boxplot", "heatmap", "dataset_boxplot"]:
        plot_types_to_generate = [args.plot_type]
    else:
        # This case should be caught by argparse choices, but good practice
        print(f"Error: Invalid plot type '{args.plot_type}' specified.")
        sys.exit(1)

    # 4. Generate Plots
    for plot_type in plot_types_to_generate:
        print(f"\n--- Generating {plot_type.replace('_', ' ').title()} ---")

        if plot_type == "boxplot":
            # --- Original F1 Boxplot Logic ---
            print(f"Generating F1 Boxplot grouped by: {args.group_by}")
            # Filter data specifically for F1 scores for this plot
            df_f1 = df_data[df_data['metric_type'] == 'f1_score'].copy()
            if df_f1.empty:
                print("Warning: No F1 score data found. Skipping F1 boxplot.")
                continue

            boxplot_args = {
                "data": df_f1,
                "group_by_column": args.group_by,
                "score_column": "metric_value", # Use the generic metric value column
                # output_path will be set below
            }
            output_filename_base = f"{args.output_filename_prefix}f1_boxplot_by_{args.group_by}"

            # Check if the specific condition for adding language hue is met
            if args.group_by == "question_model":
                print("Coloring F1 points by language for 'question_model' grouping.")
                boxplot_args["hue_column"] = "language"
                boxplot_args["hue_order"] = all_languages_list
                output_filename = f"{output_filename_base}_colored_by_language.png"
            else:
                # Default case: no hue
                output_filename = f"{output_filename_base}.png"

            output_filepath = os.path.join(args.output_dir, output_filename)
            boxplot_args["output_path"] = output_filepath

            # Call the boxplot function with the prepared arguments
            create_f1_boxplot(**boxplot_args) # Call the original F1 boxplot function


        elif plot_type == "heatmap":
            # --- Original Heatmap Logic ---
            print("\nGenerating F1 Score Heatmaps")
            # Filter data specifically for F1 scores for heatmaps
            df_f1_heatmap = df_data[df_data['metric_type'] == 'f1_score'].copy()
            if df_f1_heatmap.empty:
                print("Warning: No F1 score data found. Skipping heatmaps.")
                continue

            # --- Heatmap Type 1: Language vs Model (per Algo/Chunk/Overlap) ---
            print("\nGenerating Heatmaps: Language vs Model (per Algo/Chunk/Overlap)")
            grouping_columns_lang_model = ['retrieval_algorithm', 'chunk_size', 'overlap_size']
            # Use metric_value for the score
            required_cols_lang_model = grouping_columns_lang_model + ['language', 'question_model', 'metric_value']

            if not all(col in df_f1_heatmap.columns for col in required_cols_lang_model):
                print(f"Warning: F1 DataFrame is missing one or more required columns for Language vs Model heatmap grouping: {required_cols_lang_model}. Skipping this heatmap type.")
                # Continue to the next heatmap type if possible
            else:
                try:
                    unique_combinations_lang_model = df_f1_heatmap[grouping_columns_lang_model].drop_duplicates().to_dict('records')
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
                        filtered_df_combo = df_f1_heatmap[
                            (df_f1_heatmap['retrieval_algorithm'] == algo) &
                            (df_f1_heatmap['chunk_size'] == cs) &
                            (df_f1_heatmap['overlap_size'] == ov)
                        ].copy()

                        if filtered_df_combo.empty:
                            print("  No F1 data found for this specific combination. Skipping heatmap.")
                            continue

                        print(f"  F1 Data points for this combination: {len(filtered_df_combo)}")

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
                            values_col="metric_value", # Use generic value column
                            current_params=combo
                        )

            # --- Heatmap Type 2: Chunk Size vs Overlap Size (per Lang/Model/Algo) ---
            print("\nGenerating Heatmaps: Chunk Size vs Overlap Size (per Lang/Model/Algo)")
            grouping_columns_chunk_overlap = ['language', 'question_model', 'retrieval_algorithm']
            required_cols_chunk_overlap = grouping_columns_chunk_overlap + ['chunk_size', 'overlap_size', 'metric_value']

            if not all(col in df_f1_heatmap.columns for col in required_cols_chunk_overlap):
                 print(f"Warning: F1 DataFrame is missing one or more required columns for Chunk/Overlap heatmap grouping: {required_cols_chunk_overlap}. Skipping this heatmap type.")
                 # Continue to the next heatmap type if possible
            else:
                try:
                    unique_fixed_params_chunk = df_f1_heatmap[grouping_columns_chunk_overlap].drop_duplicates().to_dict('records')
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

                        filtered_df_chunk_combo = df_f1_heatmap[
                            (df_f1_heatmap['language'] == lang) &
                            (df_f1_heatmap['question_model'] == model) &
                            (df_f1_heatmap['retrieval_algorithm'] == algo)
                        ].copy()

                        # Check if there's enough variation to plot (at least 2x2 grid potential)
                        if filtered_df_chunk_combo.empty or \
                           filtered_df_chunk_combo['chunk_size'].nunique() < 2 or \
                           filtered_df_chunk_combo['overlap_size'].nunique() < 2:
                            print("  Skipping: Not enough F1 data or variation in chunk/overlap sizes for this combination.")
                            continue

                        print(f"  F1 Data points for this combination: {len(filtered_df_chunk_combo)}")
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
                            values_col='metric_value', # Use generic value column
                            index_col='chunk_size',
                            columns_col='overlap_size'
                        )

            # --- Heatmap Type 3: Model vs Chunk/Overlap (English Only) ---
            print("\nGenerating Heatmaps: Model vs Chunk/Overlap (English Only, per Algo)")
            required_cols_model_chunk = ['language', 'retrieval_algorithm', 'question_model', 'chunk_size', 'overlap_size', 'metric_value']

            if not all(col in df_f1_heatmap.columns for col in required_cols_model_chunk):
                 print(f"Warning: F1 DataFrame is missing one or more required columns for Model vs Chunk/Overlap heatmap: {required_cols_model_chunk}. Skipping this heatmap type.")
            else:
                # Filter for English language first
                df_english = df_f1_heatmap[df_f1_heatmap['language'] == 'english'].copy()

                if df_english.empty:
                    print("No F1 data found for language 'english'. Skipping Model vs Chunk/Overlap heatmaps.")
                else:
                    try:
                        unique_algos_english = df_english['retrieval_algorithm'].unique().tolist()
                    except KeyError:
                        print("Error: 'retrieval_algorithm' column not found in English F1 data. Skipping Model vs Chunk/Overlap heatmaps.")
                        unique_algos_english = []

                    if not unique_algos_english:
                        print("No unique retrieval algorithms found within English F1 data.")
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
                                print("  Skipping: Not enough F1 data or variation for this combination.")
                                continue

                            print(f"  F1 Data points for this combination: {len(filtered_df_model_chunk_combo)}")
                            fixed_params_model_chunk = {'language': 'english', 'retrieval_algorithm': algo}

                            sanitized_algo = sanitize_filename(str(algo))
                            fixed_param_str = f"lang_english_algo_{sanitized_algo}"
                            output_filename = f"{args.output_filename_prefix}f1_heatmap_model_vs_chunk_overlap_{fixed_param_str}.png"
                            output_filepath = os.path.join(args.output_dir, output_filename)

                            create_model_vs_chunk_overlap_heatmap(
                                data=filtered_df_model_chunk_combo,
                                output_path=output_filepath,
                                fixed_params=fixed_params_model_chunk,
                                values_col='metric_value', # Use generic value column
                                index_col_chunk='chunk_size',
                                index_col_overlap='overlap_size',
                                columns_col='question_model'
                            )

        elif plot_type == "dataset_boxplot":
            # --- New Dataset Success Boxplot Logic ---
            print(f"Generating Dataset Success Boxplots for Lang={DATASET_BOXPLOT_LANG}, Chunk={DATASET_BOXPLOT_CHUNK}, Overlap={DATASET_BOXPLOT_OVERLAP}")

            # 1. Filter data for the specific conditions
            df_filtered = df_data[
                (df_data['metric_type'] == 'dataset_success') &
                (df_data['language'] == DATASET_BOXPLOT_LANG) &
                (df_data['chunk_size'] == DATASET_BOXPLOT_CHUNK) &
                (df_data['overlap_size'] == DATASET_BOXPLOT_OVERLAP)
            ].copy() # Use copy to avoid SettingWithCopyWarning

            if df_filtered.empty:
                print(f"Warning: No dataset success data found matching the specified criteria (Lang={DATASET_BOXPLOT_LANG}, Chunk={DATASET_BOXPLOT_CHUNK}, Overlap={DATASET_BOXPLOT_OVERLAP}). Skipping dataset boxplots.")
                continue # Skip to next plot type if any

            print(f"Found {len(df_filtered)} data points matching criteria.")

            # 2. Get unique retrieval algorithms from the filtered data
            try:
                unique_algorithms = df_filtered['retrieval_algorithm'].unique().tolist()
            except KeyError:
                print("Error: 'retrieval_algorithm' column not found in filtered data. Skipping dataset boxplots.")
                continue

            if not unique_algorithms:
                print("Warning: No unique retrieval algorithms found in the filtered data. Skipping dataset boxplots.")
                continue

            print(f"Found {len(unique_algorithms)} retrieval algorithms in filtered data: {unique_algorithms}")

            # 3. Iterate through algorithms and create plots
            for i, algo in enumerate(unique_algorithms):
                print(f"\n[{i+1}/{len(unique_algorithms)}] Generating Dataset Success Boxplot for Algorithm: {algo}")

                # Filter data for the current algorithm
                df_algo_specific = df_filtered[df_filtered['retrieval_algorithm'] == algo]

                if df_algo_specific.empty:
                    print(f"  Warning: No data for algorithm '{algo}' after filtering. Skipping plot.")
                    continue

                # Check for required columns again just in case (though filtering shouldn't remove them)
                required_cols = ['question_model', 'metric_value', 'dataset_type']
                if not all(col in df_algo_specific.columns for col in required_cols):
                     print(f"  Error: Data for algorithm '{algo}' is missing required columns: {required_cols}. Skipping plot.")
                     continue

                print(f"  Data points for this algorithm: {len(df_algo_specific)}")
                print(f"  Models found: {df_algo_specific['question_model'].unique()}")
                print(f"  Dataset types found: {df_algo_specific['dataset_type'].unique()}")


                # Define output path
                sanitized_algo = sanitize_filename(algo)
                output_filename = f"{args.output_filename_prefix}dataset_success_boxplot_model_vs_dataset_{sanitized_algo}_lang_{DATASET_BOXPLOT_LANG}_cs_{DATASET_BOXPLOT_CHUNK}_os_{DATASET_BOXPLOT_OVERLAP}.png"
                output_filepath = os.path.join(args.output_dir, output_filename)

                # Call the new plotting function
                create_dataset_success_boxplot(
                    data=df_algo_specific,
                    output_path=output_filepath,
                    retrieval_algorithm=algo,
                    language=DATASET_BOXPLOT_LANG,
                    chunk_size=DATASET_BOXPLOT_CHUNK,
                    overlap_size=DATASET_BOXPLOT_OVERLAP
                    # sort_by_median_score uses default True
                )

    print("\n--- Visualization Generation Finished ---")


if __name__ == "__main__":
    main()
