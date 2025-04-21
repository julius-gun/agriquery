# visualization/plot_scripts/boxplot_generators.py
import os
import sys
import argparse # Added argparse
import pandas as pd
from typing import List, Optional, Dict, Any

# Use the utility to add paths BEFORE attempting other project imports
from plot_utils import add_project_paths, sanitize_filename
add_project_paths() # Ensure project paths are set

# Now import other modules
from visualization.visualization_data_extractor import extract_detailed_visualization_data
from visualization.plot_scripts.box_plots import create_f1_boxplot, create_dataset_success_boxplot
from utils.config_loader import ConfigLoader # Needed for language list in standalone mode


# --- Core Generator Functions ---

def generate_f1_boxplot(
    df_data: pd.DataFrame,
    group_by: str,
    output_dir: str,
    output_filename_prefix: str,
    all_languages_list: Optional[List[str]]
):
    """Generates the F1 score boxplot based on the specified grouping."""
    print(f"Generating F1 Boxplot grouped by: {group_by}")
    df_f1 = df_data[df_data['metric_type'] == 'f1_score'].copy()
    if df_f1.empty:
        print("Warning: No F1 score data found. Skipping F1 boxplot.")
        return

    boxplot_args: Dict[str, Any] = {
        "data": df_f1,
        "group_by_column": group_by,
        "score_column": "metric_value",
        "hue_column": None, # Default
        "hue_order": None, # Default
        # output_path will be set below
    }
    output_filename_base = f"{output_filename_prefix}f1_boxplot_by_{group_by}"

    # Check if the specific condition for adding language hue is met
    if group_by == "question_model":
        print("Coloring F1 points by language for 'question_model' grouping.")
        boxplot_args["hue_column"] = "language"
        boxplot_args["hue_order"] = all_languages_list
        output_filename = f"{output_filename_base}_colored_by_language.png"
    else:
        # Default case: no hue
        output_filename = f"{output_filename_base}.png"

    output_filepath = os.path.join(output_dir, output_filename)
    boxplot_args["output_path"] = output_filepath

    # Call the boxplot function with the prepared arguments
    create_f1_boxplot(**boxplot_args)


def generate_dataset_success_boxplot(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str,
    lang: str,
    chunk: int,
    overlap: int
):
    """
    Generates dataset success boxplots for a specific language, chunk size,
    and overlap size, creating one plot per retrieval algorithm found.
    """
    print(f"Generating Dataset Success Boxplots for Lang={lang}, Chunk={chunk}, Overlap={overlap}")

    # 1. Filter data for the specific conditions
    df_filtered = df_data[
        (df_data['metric_type'] == 'dataset_success') &
        (df_data['language'] == lang) &
        (df_data['chunk_size'] == chunk) &
        (df_data['overlap_size'] == overlap)
    ].copy() # Use copy to avoid SettingWithCopyWarning

    if df_filtered.empty:
        print(f"Warning: No dataset success data found matching the specified criteria (Lang={lang}, Chunk={chunk}, Overlap={overlap}). Skipping dataset boxplots.")
        return # Skip if no data matches

    print(f"Found {len(df_filtered)} data points matching criteria.")

    # 2. Get unique retrieval algorithms from the filtered data
    try:
        unique_algorithms = df_filtered['retrieval_algorithm'].unique().tolist()
    except KeyError:
        print("Error: 'retrieval_algorithm' column not found in filtered data. Skipping dataset boxplots.")
        return

    if not unique_algorithms:
        print("Warning: No unique retrieval algorithms found in the filtered data. Skipping dataset boxplots.")
        return

    print(f"Found {len(unique_algorithms)} retrieval algorithms in filtered data: {unique_algorithms}")

    # 3. Iterate through algorithms and create plots
    for i, algo in enumerate(unique_algorithms):
        print(f"\n[{i+1}/{len(unique_algorithms)}] Generating Dataset Success Boxplot for Algorithm: {algo}")

        # Filter data for the current algorithm
        df_algo_specific = df_filtered[df_filtered['retrieval_algorithm'] == algo]

        if df_algo_specific.empty:
            print(f"  Warning: No data for algorithm '{algo}' after filtering. Skipping plot.")
            continue

        # Check for required columns again just in case
        required_cols = ['question_model', 'metric_value', 'dataset_type']
        if not all(col in df_algo_specific.columns for col in required_cols):
             print(f"  Error: Data for algorithm '{algo}' is missing required columns: {required_cols}. Skipping plot.")
             continue

        print(f"  Data points for this algorithm: {len(df_algo_specific)}")
        print(f"  Models found: {df_algo_specific['question_model'].unique()}")
        print(f"  Dataset types found: {df_algo_specific['dataset_type'].unique()}")

        # Define output path
        sanitized_algo = sanitize_filename(algo) # Use imported/passed function
        output_filename = f"{output_filename_prefix}dataset_success_boxplot_model_vs_dataset_{sanitized_algo}_lang_{lang}_cs_{chunk}_os_{overlap}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        # Call the plotting function
        create_dataset_success_boxplot(
            data=df_algo_specific,
            output_path=output_filepath,
            retrieval_algorithm=algo,
            language=lang,
            chunk_size=chunk,
            overlap_size=overlap
        )

# --- Standalone Execution ---

if __name__ == "__main__":
    # Get project root and visualization dir for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(visualization_dir)

    # Define default paths relative to the project structure
    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    default_output_path = os.path.join(visualization_dir, "plots")

    parser = argparse.ArgumentParser(description="Generate specific boxplot visualizations.")
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
        "--config-path", # Needed for language list
        type=str,
        default=default_config_path,
        help="Path to the configuration JSON file (to get language list for F1 plot).",
    )
    parser.add_argument(
        "--plot-subtype",
        type=str,
        required=True,
        choices=["f1", "dataset_success"],
        help="Type of boxplot to generate.",
    )
    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="",
        help="Optional prefix for generated plot filenames.",
    )

    # Arguments specific to F1 boxplot
    parser.add_argument(
        "--group-by",
        type=str,
        default="question_model",
        choices=[
            "retrieval_algorithm", "language", "question_model",
            "chunk_size", "overlap_size", "num_retrieved_docs",
        ],
        help="Parameter to group the F1 box plot by (used if plot-subtype is 'f1').",
    )

    # Arguments specific to Dataset Success boxplot
    parser.add_argument(
        "--lang",
        type=str,
        default="english",
        help="Language to filter for dataset success boxplot.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="Chunk size to filter for dataset success boxplot.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap size to filter for dataset success boxplot.",
    )

    args = parser.parse_args()

    print("--- Starting Standalone Boxplot Generation ---")
    print(f"Plot subtype: {args.plot_subtype}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.output_filename_prefix:
        print(f"Filename prefix: {args.output_filename_prefix}")

    # 1. Load Data
    print(f"\nExtracting detailed data from: {args.results_dir}")
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        sys.exit(1) # Use sys.exit for script termination

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Generate the requested plot subtype
    if args.plot_subtype == "f1":
        # Load languages from config for consistent F1 plot coloring
        all_languages_list = None
        try:
            config_loader = ConfigLoader(args.config_path)
            language_configs = config_loader.config.get("language_configs", [])
            loaded_languages = [lc.get("language") for lc in language_configs if lc.get("language")]
            if loaded_languages:
                all_languages_list = loaded_languages
                print(f"Loaded languages for F1 plot hue order: {all_languages_list}")
            else:
                print(f"Warning: No languages found in {args.config_path}. F1 plot hue order may vary.")
        except FileNotFoundError:
            print(f"Warning: Config file not found at '{args.config_path}'. Cannot determine language list.")
        except Exception as e:
            print(f"Warning: Error loading config file '{args.config_path}': {e}.")

        generate_f1_boxplot(
            df_data=df_data,
            group_by=args.group_by,
            output_dir=args.output_dir,
            output_filename_prefix=args.output_filename_prefix,
            all_languages_list=all_languages_list # Pass loaded languages
        )
    elif args.plot_subtype == "dataset_success":
        generate_dataset_success_boxplot(
            df_data=df_data,
            output_dir=args.output_dir,
            output_filename_prefix=args.output_filename_prefix,
            lang=args.lang,
            chunk=args.chunk,
            overlap=args.overlap
        )

    print("\n--- Standalone Boxplot Generation Finished ---")