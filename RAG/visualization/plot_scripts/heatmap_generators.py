# visualization/plot_scripts/heatmap_generators.py
import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Any

# Use the utility to add paths BEFORE attempting other project imports
from plot_utils import add_project_paths, sanitize_filename
add_project_paths() # Ensure project paths are set

# Now import other modules
from visualization.visualization_data_extractor import extract_detailed_visualization_data
from visualization.plot_scripts.heatmaps import (
    create_f1_heatmap,
    create_chunk_overlap_heatmap,
    create_model_vs_chunk_overlap_heatmap
)
from utils.config_loader import ConfigLoader # Needed for language list in standalone mode

# --- Core Generator Functions ---

def generate_language_vs_model_heatmap(
    df_f1_heatmap: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str,
    all_languages_list: Optional[List[str]]
):
    """Generates F1 heatmaps: Language vs Model (per Algo/Chunk/Overlap)."""
    print("\nGenerating Heatmaps: Language vs Model (per Algo/Chunk/Overlap)")
    grouping_columns = ['retrieval_algorithm', 'chunk_size', 'overlap_size']
    required_cols = grouping_columns + ['language', 'question_model', 'metric_value']

    if not all(col in df_f1_heatmap.columns for col in required_cols):
        print(f"Warning: F1 DataFrame is missing one or more required columns for Language vs Model heatmap grouping: {required_cols}. Skipping this heatmap type.")
        return

    try:
        unique_combinations = df_f1_heatmap[grouping_columns].drop_duplicates().to_dict('records')
    except KeyError as e:
        print(f"Error: Could not find grouping columns {grouping_columns} for Language vs Model heatmaps. Error: {e}. Skipping this heatmap type.")
        return

    if not unique_combinations:
        print("No unique combinations of (retrieval_algorithm, chunk_size, overlap_size) found for Language vs Model heatmaps.")
        return

    print(f"Found {len(unique_combinations)} unique parameter combinations for Language vs Model heatmaps.")

    for i, combo in enumerate(unique_combinations):
        algo = combo['retrieval_algorithm']
        cs = combo['chunk_size']
        ov = combo['overlap_size']

        print(f"\n[{i+1}/{len(unique_combinations)}] Generating Language vs Model Heatmap for: Algo={algo}, Chunk={cs}, Overlap={ov}")

        filtered_df_combo = df_f1_heatmap[
            (df_f1_heatmap['retrieval_algorithm'] == algo) &
            (df_f1_heatmap['chunk_size'] == cs) &
            (df_f1_heatmap['overlap_size'] == ov)
        ].copy()

        if filtered_df_combo.empty:
            print("  No F1 data found for this specific combination. Skipping heatmap.")
            continue

        print(f"  F1 Data points for this combination: {len(filtered_df_combo)}")

        combo_str = f"algo_{sanitize_filename(str(algo))}_cs_{cs}_os_{ov}"
        output_filename = f"{output_filename_prefix}f1_heatmap_lang_vs_model_{combo_str}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        create_f1_heatmap(
            data=filtered_df_combo,
            output_path=output_filepath,
            all_indices=all_languages_list, # Pass the full language list
            index_col="language",
            columns_col="question_model",
            values_col="metric_value", # The actual data column name
            value_label="F1 Score",   # NEW: Explicit label for the title
            current_params=combo
        )

def generate_chunk_vs_overlap_heatmap(
    df_f1_heatmap: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str
):
    """Generates F1 heatmaps: Chunk Size vs Overlap Size (per Lang/Model/Algo)."""
    print("\nGenerating Heatmaps: Chunk Size vs Overlap Size (per Lang/Model/Algo)")
    grouping_columns = ['language', 'question_model', 'retrieval_algorithm']
    required_cols = grouping_columns + ['chunk_size', 'overlap_size', 'metric_value']

    if not all(col in df_f1_heatmap.columns for col in required_cols):
         print(f"Warning: F1 DataFrame is missing one or more required columns for Chunk/Overlap heatmap grouping: {required_cols}. Skipping this heatmap type.")
         return

    try:
        unique_fixed_params = df_f1_heatmap[grouping_columns].drop_duplicates().to_dict('records')
    except KeyError as e:
        print(f"Error: Could not find grouping columns {grouping_columns} for Chunk/Overlap heatmaps. Error: {e}. Skipping this heatmap type.")
        return

    if not unique_fixed_params:
        print("No unique combinations of (language, question_model, retrieval_algorithm) found for Chunk/Overlap heatmaps.")
        return

    print(f"Found {len(unique_fixed_params)} unique fixed parameter combinations for Chunk/Overlap heatmaps.")
    for i, fixed_params in enumerate(unique_fixed_params):
        lang = fixed_params['language']
        model = fixed_params['question_model']
        algo = fixed_params['retrieval_algorithm']
        print(f"\n[{i+1}/{len(unique_fixed_params)}] Generating Chunk/Overlap Heatmap for: Lang={lang}, Model={model}, Algo={algo}")

        filtered_df_chunk_combo = df_f1_heatmap[
            (df_f1_heatmap['language'] == lang) &
            (df_f1_heatmap['question_model'] == model) &
            (df_f1_heatmap['retrieval_algorithm'] == algo)
        ].copy()

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
        output_filename = f"{output_filename_prefix}f1_heatmap_chunk_vs_overlap_{fixed_param_str}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        create_chunk_overlap_heatmap(
            data=filtered_df_chunk_combo,
            output_path=output_filepath,
            fixed_params=fixed_params,
            values_col='metric_value', # The actual data column name
            value_label="F1 Score",   # NEW: Explicit label for the title
            index_col='chunk_size',
            columns_col='overlap_size'
        )

def generate_model_vs_chunk_overlap_heatmap(
    df_f1_heatmap: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str
):
    """Generates F1 heatmaps: Model vs Chunk/Overlap (English Only, per Algo)."""
    print("\nGenerating Heatmaps: Model vs Chunk/Overlap (English Only, per Algo)")
    required_cols = ['language', 'retrieval_algorithm', 'question_model', 'chunk_size', 'overlap_size', 'metric_value']

    if not all(col in df_f1_heatmap.columns for col in required_cols):
         print(f"Warning: F1 DataFrame is missing one or more required columns for Model vs Chunk/Overlap heatmap: {required_cols}. Skipping this heatmap type.")
         return

    df_english = df_f1_heatmap[df_f1_heatmap['language'] == 'english'].copy()

    if df_english.empty:
        print("No F1 data found for language 'english'. Skipping Model vs Chunk/Overlap heatmaps.")
        return

    try:
        unique_algos_english = df_english['retrieval_algorithm'].unique().tolist()
    except KeyError:
        print("Error: 'retrieval_algorithm' column not found in English F1 data. Skipping Model vs Chunk/Overlap heatmaps.")
        return

    if not unique_algos_english:
        print("No unique retrieval algorithms found within English F1 data.")
        return

    print(f"Found {len(unique_algos_english)} algorithms for English Model vs Chunk/Overlap heatmaps: {unique_algos_english}")

    for i, algo in enumerate(unique_algos_english):
        print(f"\n[{i+1}/{len(unique_algos_english)}] Generating Model vs Chunk/Overlap Heatmap for: Lang=english, Algo={algo}")

        filtered_df_model_chunk_combo = df_english[
            df_english['retrieval_algorithm'] == algo
        ].copy()

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
        output_filename = f"{output_filename_prefix}f1_heatmap_model_vs_chunk_overlap_{fixed_param_str}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        create_model_vs_chunk_overlap_heatmap(
            data=filtered_df_model_chunk_combo,
            output_path=output_filepath,
            fixed_params=fixed_params_model_chunk,
            values_col='metric_value',       # The actual data column name
            value_label="F1 Score",         # NEW: Explicit label for the title
            index_col_chunk='chunk_size',
            index_col_overlap='overlap_size',
            columns_col='question_model'
        )
def generate_dataset_success_heatmaps(
    df_data: pd.DataFrame, # Takes the full dataframe
    output_dir: str,
    output_filename_prefix: str
):
    """
    Generates heatmaps for Dataset Success Rate: Model vs Chunk/Overlap
    (English Only, per Algo/Dataset Type). Excludes 'zeroshot' algorithm
    as it lacks chunk/overlap parameters.
    """
    print("\nGenerating Heatmaps: Dataset Success Rate - Model vs Chunk/Overlap (English Only, per Algo/Dataset)")
    target_language = 'english'
    target_metric = 'dataset_success'
    required_cols = [
        'language', 'metric_type', 'retrieval_algorithm', 'dataset_type',
        'question_model', 'chunk_size', 'overlap_size', 'metric_value'
    ]

    if not all(col in df_data.columns for col in required_cols):
         print(f"Warning: Input DataFrame is missing one or more required columns for Dataset Success heatmaps: {required_cols}. Skipping this heatmap type.")
         return

    # Filter for English language and dataset_success metric
    df_filtered = df_data[
        (df_data['language'] == target_language) &
        (df_data['metric_type'] == target_metric)
    ].copy()

    if df_filtered.empty:
        print(f"No data found for language '{target_language}' and metric '{target_metric}'. Skipping Dataset Success heatmaps.")
        return

    # Get unique algorithms and dataset types from the filtered data
    try:
        # --- MODIFICATION START ---
        # Filter out 'zeroshot' before getting unique algorithms for this specific plot type
        df_filtered_non_zeroshot = df_filtered[df_filtered['retrieval_algorithm'] != 'zeroshot']
        if df_filtered_non_zeroshot.empty:
             print(f"No non-zeroshot data found for language '{target_language}' and metric '{target_metric}'. Skipping these heatmaps.")
             return

        unique_algos = df_filtered_non_zeroshot['retrieval_algorithm'].unique().tolist()
        # --- MODIFICATION END ---

        # Get unique datasets from the original filtered data (might include zeroshot datasets if needed elsewhere later)
        unique_datasets = df_filtered['dataset_type'].unique().tolist()
    except KeyError as e:
        print(f"Error: Missing expected column '{e}' in filtered data. Skipping Dataset Success heatmaps.")
        return

    if not unique_algos:
        print("No unique non-zeroshot retrieval algorithms found in the filtered data.")
        return
    if not unique_datasets:
        print("No unique dataset types found in the filtered data.")
        return

    # --- MODIFICATION: Updated print statement ---
    print(f"Found {len(unique_algos)} non-zeroshot algorithms and {len(unique_datasets)} dataset types for English Dataset Success heatmaps.")
    print(f"  Algorithms (excluding zeroshot): {unique_algos}")
    # --- End Modification ---
    print(f"  Dataset Types: {unique_datasets}")

    total_plots = len(unique_algos) * len(unique_datasets)
    plot_counter = 0

    # Iterate through each non-zeroshot algorithm and dataset type
    for algo in unique_algos: # This loop now only contains non-zeroshot algos
        for dataset in unique_datasets:
            plot_counter += 1
            print(f"\n[{plot_counter}/{total_plots}] Generating Dataset Success Heatmap for: Lang={target_language}, Algo={algo}, Dataset={dataset}")

            # Filter further for the specific algorithm and dataset type
            # Use df_filtered_non_zeroshot here as well to ensure consistency
            df_combo = df_filtered_non_zeroshot[
                (df_filtered_non_zeroshot['retrieval_algorithm'] == algo) &
                (df_filtered_non_zeroshot['dataset_type'] == dataset)
            ].copy()

            # Check if data exists and has variation for this specific combination
            # Need to check for NaN/None in chunk/overlap as well, although filtering zeroshot should handle it
            if df_combo.empty or \
               df_combo['chunk_size'].isnull().all() or \
               df_combo['overlap_size'].isnull().all() or \
               df_combo['chunk_size'].nunique() < 1 or \
               df_combo['overlap_size'].nunique() < 1 or \
               df_combo['question_model'].nunique() < 1:
                print("  Skipping: Not enough data or variation (models, chunk/overlap) for this specific combination.")
                continue

            print(f"  Data points for this combination: {len(df_combo)}")

            # Prepare parameters for the plotting function
            fixed_params_combo = {
                'language': target_language,
                'retrieval_algorithm': algo,
                'dataset_type': dataset
            }

            # Construct filename
            sanitized_algo = sanitize_filename(str(algo))
            sanitized_dataset = sanitize_filename(str(dataset))
            fixed_param_str = f"lang_{target_language}_algo_{sanitized_algo}_dataset_{sanitized_dataset}"
            output_filename = f"{output_filename_prefix}dataset_success_heatmap_model_vs_chunk_overlap_{fixed_param_str}.png"
            output_filepath = os.path.join(output_dir, output_filename)

            # Call the existing heatmap function
            create_model_vs_chunk_overlap_heatmap(
                data=df_combo,
                output_path=output_filepath,
                fixed_params=fixed_params_combo,
                values_col='metric_value',       # The actual data column name
                value_label="Success Rate",     # Specific label for this metric
                index_col_chunk='chunk_size',
                index_col_overlap='overlap_size',
                columns_col='question_model',
                sort_columns_by_value=True # Keep sorting models by performance
            )

def generate_algo_vs_model_f1_heatmap(
    df_data: pd.DataFrame, # Takes the full dataframe
    output_dir: str,
    output_filename_prefix: str
):
    """
    Generates a single heatmap comparing Retrieval Algorithms vs LLM Models based on F1 Score,
    specifically for the 'english' language.
    Averages F1 scores across all other parameters (chunk, overlap, etc.) within English results.
    """
    # --- MODIFICATION START ---
    target_language = 'english'
    print(f"\nGenerating Heatmap: F1 Score - Algorithm vs Model (Language: {target_language})")
    metric_to_plot = 'f1_score'
    # Add 'language' to required columns
    required_cols = ['retrieval_algorithm', 'question_model', 'metric_type', 'metric_value', 'language']
    # --- MODIFICATION END ---

    if not all(col in df_data.columns for col in required_cols):
         print(f"Warning: Input DataFrame is missing one or more required columns for Algo vs Model F1 ({target_language}) heatmap: {required_cols}. Skipping.")
         return

    # 1. Filter for the relevant metric AND language
    df_filtered = df_data[
        (df_data['metric_type'] == metric_to_plot) &
        (df_data['language'] == target_language)
    ].copy()

    if df_filtered.empty:
        # --- MODIFICATION: Updated message ---
        print(f"No data found for metric_type '{metric_to_plot}' and language '{target_language}'. Skipping Algo vs Model F1 heatmap.")
        # --- MODIFICATION END ---
        return

    # 2. Aggregate data: Mean F1 score per Algo/Model combination (for English)
    try:
        df_agg = df_filtered.groupby(['retrieval_algorithm', 'question_model'])['metric_value'].mean().reset_index()
        print(f"  Aggregated F1 data points ({target_language} only): {len(df_agg)}")
        if df_agg.empty:
             print("  Aggregation resulted in empty DataFrame. Skipping plot.")
             return
    except KeyError as e:
        # --- MODIFICATION: Updated message ---
        print(f"Error during aggregation for Algo vs Model F1 ({target_language}) heatmap: Missing column {e}. Skipping.")
        # --- MODIFICATION END ---
        return
    except Exception as e:
        # --- MODIFICATION: Updated message ---
        print(f"Error during aggregation for Algo vs Model F1 ({target_language}) heatmap: {e}. Skipping.")
        # --- MODIFICATION END ---
        return

    # 3. Prepare for plotting
    # --- MODIFICATION: Add language to filename ---
    output_filename = f"{output_filename_prefix}f1_heatmap_algo_vs_model_{target_language}.png"
    # --- MODIFICATION END ---
    output_filepath = os.path.join(output_dir, output_filename)

    # 4. Call the generic heatmap function
    create_f1_heatmap( # Re-using the existing function
        data=df_agg,
        output_path=output_filepath,
        index_col="retrieval_algorithm", # y-axis
        columns_col="question_model",    # x-axis
        values_col="metric_value",       # The aggregated mean F1 score
        value_label="Mean F1 Score",     # Label for the title/colorbar
        all_indices=None,                # Let it use indices present in aggregated data
        current_params=None,             # No specific sub-parameters for this plot
        # --- MODIFICATION: Update title ---
        title=f"Mean F1 Score (English): Retrieval Algorithm vs LLM Model", # Custom title
        # --- MODIFICATION END ---
        sort_columns_by_value=True       # Sort models by performance
    )

# --- MODIFIED: Generator for Algo vs Model Mean Dataset Success Heatmap (English Only) ---
def generate_algo_vs_model_dataset_success_heatmap(
    df_data: pd.DataFrame, # Takes the full dataframe
    output_dir: str,
    output_filename_prefix: str
):
    """
    Generates separate heatmaps for each specified dataset type comparing Algorithms
    (RAG filtered by specific chunk/overlap, ZeroShot by specific noise levels)
    vs LLM Models based on Dataset Success Rate for the 'english' language.
    """
    target_language = 'english'
    metric_to_plot = 'dataset_success'
    # Define target datasets based on config keys (adjust if keys differ)
    target_datasets = ["general_questions", "table_questions", "unanswerable_questions"]
    target_chunk = 200
    target_overlap = 100
    target_noise_levels = [1000, 10000, 30000, 59000]
    # Define RAG algos explicitly or filter dynamically (using != 'zeroshot' is simpler)
    # rag_algorithms = ['keyword', 'hybrid', 'embedding']

    print(f"\nGenerating Dataset Success Heatmaps: Algorithm vs Model (Language: {target_language})")
    print(f"  RAG Params: Chunk={target_chunk}, Overlap={target_overlap}")
    print(f"  ZeroShot Noise Levels: {target_noise_levels}")
    print(f"  Target Datasets: {target_datasets}")

    required_cols = [
        'retrieval_algorithm', 'question_model', 'metric_type', 'metric_value',
        'dataset_type', 'language', 'chunk_size', 'overlap_size', 'noise_level'
    ]
    # Check if all required columns are present
    missing_cols = [col for col in required_cols if col not in df_data.columns]
    if missing_cols:
         print(f"Warning: Input DataFrame is missing one or more required columns: {missing_cols}. Skipping these heatmaps.")
         return

    # Initial filter for language and metric
    df_lang_metric_filtered = df_data[
        (df_data['language'] == target_language) &
        (df_data['metric_type'] == metric_to_plot)
    ].copy()

    if df_lang_metric_filtered.empty:
        print(f"No data found for metric '{metric_to_plot}' and language '{target_language}'. Skipping.")
        return

    # Convert relevant columns to numeric, coercing errors to NaN for safe comparison/filtering
    numeric_cols = ['chunk_size', 'overlap_size', 'noise_level']
    for col in numeric_cols:
        if col in df_lang_metric_filtered.columns:
            df_lang_metric_filtered[col] = pd.to_numeric(df_lang_metric_filtered[col], errors='coerce')

    plot_counter = 0
    for dataset_type in target_datasets:
        plot_counter += 1
        print(f"\n[{plot_counter}/{len(target_datasets)}] Processing dataset: {dataset_type}")

        # Filter for the current dataset
        df_dataset_filtered = df_lang_metric_filtered[
            df_lang_metric_filtered['dataset_type'] == dataset_type
        ].copy()

        if df_dataset_filtered.empty:
            print(f"  No data found for dataset '{dataset_type}'. Skipping heatmap.")
            continue

        # --- Filter RAG Data ---
        # Identify RAG rows (not 'zeroshot')
        is_rag = df_dataset_filtered['retrieval_algorithm'] != 'zeroshot'

        # Apply RAG filters (chunk, overlap)
        # Comparisons handle NaN correctly (NaN == value is False)
        df_rag_filtered = df_dataset_filtered[
            is_rag &
            (df_dataset_filtered['chunk_size'] == target_chunk) &
            (df_dataset_filtered['overlap_size'] == target_overlap)
        ].copy()

        # Add plot index for RAG
        if not df_rag_filtered.empty:
             df_rag_filtered['plot_index'] = df_rag_filtered['retrieval_algorithm']
             print(f"  Found {len(df_rag_filtered)} RAG data points matching criteria.")
        else:
             print("  No RAG data points found matching criteria.")


        # --- Filter ZeroShot Data ---
        # Apply ZeroShot filters (algorithm name, noise levels)
        df_zeroshot_filtered = df_dataset_filtered[
            (df_dataset_filtered['retrieval_algorithm'] == 'zeroshot') &
            (df_dataset_filtered['noise_level'].isin(target_noise_levels))
        ].copy()

        # Add plot index for ZeroShot
        if not df_zeroshot_filtered.empty:
            # Create index like 'zeroshot_noise_1000'
            # Ensure noise_level is integer for formatting, fill potential NaNs from coerce step if any survived isin
            df_zeroshot_filtered['noise_level_int'] = df_zeroshot_filtered['noise_level'].fillna(0).astype(int)
            df_zeroshot_filtered['plot_index'] = 'zeroshot_noise_' + df_zeroshot_filtered['noise_level_int'].astype(str)
            # Drop the temporary column if desired
            df_zeroshot_filtered = df_zeroshot_filtered.drop(columns=['noise_level_int'])
            print(f"  Found {len(df_zeroshot_filtered)} ZeroShot data points matching criteria.")
        else:
             print("  No ZeroShot data points found matching criteria.")


        # --- Combine and Prepare for Plotting ---
        # Use list comprehension to handle cases where one df might be empty
        dfs_to_concat = [df for df in [df_rag_filtered, df_zeroshot_filtered] if not df.empty]

        if not dfs_to_concat:
             print(f"  No data remaining for dataset '{dataset_type}' after applying RAG/ZeroShot filters. Skipping heatmap.")
             continue

        df_combined = pd.concat(dfs_to_concat, ignore_index=True)

        # Check for necessary columns again after potential filtering/concatenation issues
        if 'plot_index' not in df_combined.columns or 'question_model' not in df_combined.columns or 'metric_value' not in df_combined.columns:
             print(f"  Error: Missing essential columns ('plot_index', 'question_model', 'metric_value') in combined data for {dataset_type}. Skipping heatmap.")
             continue

        print(f"  Total data points for heatmap: {len(df_combined)}")
        print(f"  Unique indices found: {df_combined['plot_index'].unique().tolist()}")
        print(f"  Unique models found: {df_combined['question_model'].unique().tolist()}")


        # --- Plotting ---
        sanitized_dataset_type = sanitize_filename(dataset_type) # Sanitize dataset name for filename
        output_filename = f"{output_filename_prefix}dataset_success_heatmap_algo_vs_model_{target_language}_{sanitized_dataset_type}.png"
        output_filepath = os.path.join(output_dir, output_filename)
        plot_title = (
            f"{dataset_type.replace('_', ' ').title()} Success Rate ({target_language.title()}):\n"
            f"Algorithm (RAG: C={target_chunk}/O={target_overlap}, ZeroShot: Noise) vs LLM Model"
        )

        # Call the generic heatmap function (which handles pivoting)
        create_f1_heatmap( # Re-using the existing function
            data=df_combined,             # Pass the combined, filtered, UNPIVOTED data
            output_path=output_filepath,
            index_col="plot_index",       # Use the new combined index (Y-axis: Algos/Zeroshot variants)
            columns_col="question_model", # X-axis: Models
            values_col="metric_value",    # Cell values: Success Rate
            value_label="Success Rate",   # Label for the title/colorbar
            all_indices=None,             # Let it determine indices from data
            current_params=None,          # No specific sub-parameters for this plot type
            title=plot_title,             # Custom title reflecting dataset and filters
            sort_columns_by_value=True    # Sort models by performance
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

    parser = argparse.ArgumentParser(description="Generate specific F1 heatmap visualizations.")
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
        help="Path to the configuration JSON file (to get language list for lang_vs_model heatmap).",
    )
    parser.add_argument(
        "--plot-subtype",
        type=str,
        required=True,
        # --- MODIFICATION START: Added new choices ---
        choices=[
            "lang_vs_model",
            "chunk_vs_overlap",
            "model_vs_chunk_overlap",
            "dataset_success", # This is the Model vs Chunk/Overlap for Dataset Success
            "algo_vs_model_f1", # New choice for F1 Algo vs Model
            "algo_vs_model_success", # New choice for Mean Success Algo vs Model (now per dataset)
            "all"
        ],
        # --- MODIFICATION END ---
        help="Type of heatmap(s) to generate.",
    )
    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="",
        help="Optional prefix for generated plot filenames.",
    )

    args = parser.parse_args()

    print("--- Starting Standalone Heatmap Generation ---")
    print(f"Plot subtype(s): {args.plot_subtype}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.output_filename_prefix:
        print(f"Filename prefix: {args.output_filename_prefix}")

    # 1. Load Data
    print(f"\nExtracting detailed data from: {args.results_dir}")
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load languages from config (needed for lang_vs_model)
    all_languages_list = None
    if args.plot_subtype in ["lang_vs_model", "all"]:
        try:
            config_loader = ConfigLoader(args.config_path)
            language_configs = config_loader.config.get("language_configs", [])
            loaded_languages = [lc.get("language") for lc in language_configs if lc.get("language")]
            if loaded_languages:
                all_languages_list = loaded_languages
                print(f"Loaded languages for heatmap axes/order: {all_languages_list}")
            else:
                print(f"Warning: No languages found in {args.config_path}. Heatmap axes order may vary.")
        except FileNotFoundError:
            print(f"Warning: Config file not found at '{args.config_path}'. Cannot determine language list.")
        except Exception as e:
            print(f"Warning: Error loading config file '{args.config_path}': {e}.")

    # 2. Generate the requested plot subtype(s)
    subtypes_to_generate = []
    if args.plot_subtype == "all":
        # --- MODIFICATION START: Added new types to 'all' ---
        subtypes_to_generate = [
            "lang_vs_model",
            "chunk_vs_overlap",
            "model_vs_chunk_overlap",
            "dataset_success",
            "algo_vs_model_f1",
            "algo_vs_model_success"
        ]
        # --- MODIFICATION END ---
    else:
        subtypes_to_generate = [args.plot_subtype]

    for subtype in subtypes_to_generate:
        if subtype == "lang_vs_model":
            # This one specifically needs F1 data
            df_f1_heatmap = df_data[df_data['metric_type'] == 'f1_score'].copy()
            if df_f1_heatmap.empty:
                print("Warning: No F1 score data found. Skipping lang_vs_model heatmap.")
                continue
            generate_language_vs_model_heatmap(
                df_f1_heatmap=df_f1_heatmap,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                all_languages_list=all_languages_list
            )
        elif subtype == "chunk_vs_overlap":
             # This one specifically needs F1 data
            df_f1_heatmap = df_data[df_data['metric_type'] == 'f1_score'].copy()
            if df_f1_heatmap.empty:
                print("Warning: No F1 score data found. Skipping chunk_vs_overlap heatmap.")
                continue
            generate_chunk_vs_overlap_heatmap(
                df_f1_heatmap=df_f1_heatmap,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )
        elif subtype == "model_vs_chunk_overlap":
             # This one specifically needs F1 data
            df_f1_heatmap = df_data[df_data['metric_type'] == 'f1_score'].copy()
            if df_f1_heatmap.empty:
                print("Warning: No F1 score data found. Skipping model_vs_chunk_overlap heatmap.")
                continue
            generate_model_vs_chunk_overlap_heatmap(
                df_f1_heatmap=df_f1_heatmap,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )
        elif subtype == "dataset_success": # This is the Model vs Chunk/Overlap for Dataset Success
            generate_dataset_success_heatmaps(
                df_data=df_data, # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )
        # --- NEW: Call the new generator functions ---
        elif subtype == "algo_vs_model_f1":
            generate_algo_vs_model_f1_heatmap(
                df_data=df_data, # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )
        elif subtype == "algo_vs_model_success":
            generate_algo_vs_model_dataset_success_heatmap(
                df_data=df_data, # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix
            )

    print("\n--- Standalone Heatmap Generation Finished ---")