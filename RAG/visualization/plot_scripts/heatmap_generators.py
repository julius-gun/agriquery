# visualization/plot_scripts/heatmap_generators.py
import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Any

# Use the utility to add paths BEFORE attempting other project imports
from plot_utils import add_project_paths, sanitize_filename

add_project_paths()  # Ensure project paths are set

# Now import other modules
from visualization.visualization_data_extractor import (
    extract_detailed_visualization_data,
)
from visualization.plot_scripts.heatmaps import (
    create_f1_heatmap,
    create_chunk_overlap_heatmap,
    create_model_vs_chunk_overlap_heatmap,
)
from utils.config_loader import (
    ConfigLoader,
)  # Needed for language list in standalone mode

# --- Core Generator Functions ---


def generate_language_vs_model_heatmap(
    df_f1_heatmap: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str,
    all_languages_list: Optional[List[str]],
):
    """Generates F1 heatmaps: Language vs Model (per Algo/Chunk/Overlap)."""
    print("\nGenerating Heatmaps: Language vs Model (per Algo/Chunk/Overlap)")
    grouping_columns = ["retrieval_algorithm", "chunk_size", "overlap_size"]
    required_cols = grouping_columns + ["language", "question_model", "metric_value"]

    if not all(col in df_f1_heatmap.columns for col in required_cols):
        print(
            f"Warning: F1 DataFrame is missing one or more required columns for Language vs Model heatmap grouping: {required_cols}. Skipping this heatmap type."
        )
        return

    try:
        unique_combinations = (
            df_f1_heatmap[grouping_columns].drop_duplicates().to_dict("records")
        )
    except KeyError as e:
        print(
            f"Error: Could not find grouping columns {grouping_columns} for Language vs Model heatmaps. Error: {e}. Skipping this heatmap type."
        )
        return

    if not unique_combinations:
        print(
            "No unique combinations of (retrieval_algorithm, chunk_size, overlap_size) found for Language vs Model heatmaps."
        )
        return

    print(
        f"Found {len(unique_combinations)} unique parameter combinations for Language vs Model heatmaps."
    )

    for i, combo in enumerate(unique_combinations):
        algo = combo["retrieval_algorithm"]
        cs = combo["chunk_size"]
        ov = combo["overlap_size"]

        print(
            f"\n[{i + 1}/{len(unique_combinations)}] Generating Language vs Model Heatmap for: Algo={algo}, Chunk={cs}, Overlap={ov}"
        )

        filtered_df_combo = df_f1_heatmap[
            (df_f1_heatmap["retrieval_algorithm"] == algo)
            & (df_f1_heatmap["chunk_size"] == cs)
            & (df_f1_heatmap["overlap_size"] == ov)
        ].copy()

        if filtered_df_combo.empty:
            print("  No F1 data found for this specific combination. Skipping heatmap.")
            continue

        print(f"  F1 Data points for this combination: {len(filtered_df_combo)}")

        combo_str = f"algo_{sanitize_filename(str(algo))}_cs_{cs}_os_{ov}"
        output_filename = (
            f"{output_filename_prefix}f1_heatmap_lang_vs_model_{combo_str}.png"
        )
        output_filepath = os.path.join(output_dir, output_filename)

        create_f1_heatmap(
            data=filtered_df_combo,
            output_path=output_filepath,
            all_indices=all_languages_list,  # Pass the full language list
            index_col="language",
            columns_col="question_model",
            values_col="metric_value",  # The actual data column name
            value_label="F1 Score",  # NEW: Explicit label for the title
            current_params=combo,
        )


def generate_chunk_vs_overlap_heatmap(
    df_f1_heatmap: pd.DataFrame, output_dir: str, output_filename_prefix: str
):
    """Generates F1 heatmaps: Chunk Size vs Overlap Size (per Lang/Model/Algo)."""
    print("\nGenerating Heatmaps: Chunk Size vs Overlap Size (per Lang/Model/Algo)")
    grouping_columns = ["language", "question_model", "retrieval_algorithm"]
    required_cols = grouping_columns + ["chunk_size", "overlap_size", "metric_value"]

    if not all(col in df_f1_heatmap.columns for col in required_cols):
        print(
            f"Warning: F1 DataFrame is missing one or more required columns for Chunk/Overlap heatmap grouping: {required_cols}. Skipping this heatmap type."
        )
        return

    try:
        unique_fixed_params = (
            df_f1_heatmap[grouping_columns].drop_duplicates().to_dict("records")
        )
    except KeyError as e:
        print(
            f"Error: Could not find grouping columns {grouping_columns} for Chunk/Overlap heatmaps. Error: {e}. Skipping this heatmap type."
        )
        return

    if not unique_fixed_params:
        print(
            "No unique combinations of (language, question_model, retrieval_algorithm) found for Chunk/Overlap heatmaps."
        )
        return

    print(
        f"Found {len(unique_fixed_params)} unique fixed parameter combinations for Chunk/Overlap heatmaps."
    )
    for i, fixed_params in enumerate(unique_fixed_params):
        lang = fixed_params["language"]
        model = fixed_params["question_model"]
        algo = fixed_params["retrieval_algorithm"]
        print(
            f"\n[{i + 1}/{len(unique_fixed_params)}] Generating Chunk/Overlap Heatmap for: Lang={lang}, Model={model}, Algo={algo}"
        )

        filtered_df_chunk_combo = df_f1_heatmap[
            (df_f1_heatmap["language"] == lang)
            & (df_f1_heatmap["question_model"] == model)
            & (df_f1_heatmap["retrieval_algorithm"] == algo)
        ].copy()

        if (
            filtered_df_chunk_combo.empty
            or filtered_df_chunk_combo["chunk_size"].nunique() < 2
            or filtered_df_chunk_combo["overlap_size"].nunique() < 2
        ):
            print(
                "  Skipping: Not enough F1 data or variation in chunk/overlap sizes for this combination."
            )
            continue

        print(f"  F1 Data points for this combination: {len(filtered_df_chunk_combo)}")
        sanitized_model = sanitize_filename(str(model))
        sanitized_algo = sanitize_filename(str(algo))
        sanitized_lang = sanitize_filename(str(lang))
        fixed_param_str = (
            f"lang_{sanitized_lang}_model_{sanitized_model}_algo_{sanitized_algo}"
        )
        output_filename = (
            f"{output_filename_prefix}f1_heatmap_chunk_vs_overlap_{fixed_param_str}.png"
        )
        output_filepath = os.path.join(output_dir, output_filename)

        create_chunk_overlap_heatmap(
            data=filtered_df_chunk_combo,
            output_path=output_filepath,
            fixed_params=fixed_params,
            values_col="metric_value",  # The actual data column name
            value_label="F1 Score",  # NEW: Explicit label for the title
            index_col="chunk_size",
            columns_col="overlap_size",
        )


def generate_model_vs_chunk_overlap_heatmap(
    df_f1_heatmap: pd.DataFrame, output_dir: str, output_filename_prefix: str
):
    """Generates F1 heatmaps: Model vs Chunk/Overlap (English Only, per Algo)."""
    print("\nGenerating Heatmaps: Model vs Chunk/Overlap (English Only, per Algo)")
    required_cols = [
        "language",
        "retrieval_algorithm",
        "question_model",
        "chunk_size",
        "overlap_size",
        "metric_value",
    ]

    if not all(col in df_f1_heatmap.columns for col in required_cols):
        print(
            f"Warning: F1 DataFrame is missing one or more required columns for Model vs Chunk/Overlap heatmap: {required_cols}. Skipping this heatmap type."
        )
        return

    df_english = df_f1_heatmap[df_f1_heatmap["language"] == "english"].copy()

    if df_english.empty:
        print(
            "No F1 data found for language 'english'. Skipping Model vs Chunk/Overlap heatmaps."
        )
        return

    try:
        unique_algos_english = df_english["retrieval_algorithm"].unique().tolist()
    except KeyError:
        print(
            "Error: 'retrieval_algorithm' column not found in English F1 data. Skipping Model vs Chunk/Overlap heatmaps."
        )
        return

    if not unique_algos_english:
        print("No unique retrieval algorithms found within English F1 data.")
        return

    print(
        f"Found {len(unique_algos_english)} algorithms for English Model vs Chunk/Overlap heatmaps: {unique_algos_english}"
    )

    for i, algo in enumerate(unique_algos_english):
        print(
            f"\n[{i + 1}/{len(unique_algos_english)}] Generating Model vs Chunk/Overlap Heatmap for: Lang=english, Algo={algo}"
        )

        filtered_df_model_chunk_combo = df_english[
            df_english["retrieval_algorithm"] == algo
        ].copy()

        if (
            filtered_df_model_chunk_combo.empty
            or filtered_df_model_chunk_combo["chunk_size"].nunique() < 1
            or filtered_df_model_chunk_combo["overlap_size"].nunique() < 1
            or filtered_df_model_chunk_combo["question_model"].nunique() < 1
        ):
            print("  Skipping: Not enough F1 data or variation for this combination.")
            continue

        print(
            f"  F1 Data points for this combination: {len(filtered_df_model_chunk_combo)}"
        )
        fixed_params_model_chunk = {"language": "english", "retrieval_algorithm": algo}

        sanitized_algo = sanitize_filename(str(algo))
        fixed_param_str = f"lang_english_algo_{sanitized_algo}"
        output_filename = f"{output_filename_prefix}f1_heatmap_model_vs_chunk_overlap_{fixed_param_str}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        create_model_vs_chunk_overlap_heatmap(
            data=filtered_df_model_chunk_combo,
            output_path=output_filepath,
            fixed_params=fixed_params_model_chunk,
            values_col="metric_value",  # The actual data column name
            value_label="F1 Score",  # NEW: Explicit label for the title
            index_col_chunk="chunk_size",
            index_col_overlap="overlap_size",
            columns_col="question_model",
        )


def generate_dataset_success_heatmaps(
    df_data: pd.DataFrame,  # Takes the full dataframe
    output_dir: str,
    output_filename_prefix: str,
):
    """
    Generates heatmaps for Dataset Success Rate: Model vs Chunk/Overlap
    (English Only, per Algo/Dataset Type). Excludes 'zeroshot' algorithm
    as it lacks chunk/overlap parameters.
    """
    print(
        "\nGenerating Heatmaps: Dataset Success Rate - Model vs Chunk/Overlap (English Only, per Algo/Dataset)"
    )
    target_language = "english"
    target_metric = "dataset_success"
    required_cols = [
        "language",
        "metric_type",
        "retrieval_algorithm",
        "dataset_type",
        "question_model",
        "chunk_size",
        "overlap_size",
        "metric_value",
    ]

    if not all(col in df_data.columns for col in required_cols):
        print(
            f"Warning: Input DataFrame is missing one or more required columns for Dataset Success heatmaps: {required_cols}. Skipping this heatmap type."
        )
        return

    # Filter for English language and dataset_success metric
    df_filtered = df_data[
        (df_data["language"] == target_language)
        & (df_data["metric_type"] == target_metric)
    ].copy()

    if df_filtered.empty:
        print(
            f"No data found for language '{target_language}' and metric '{target_metric}'. Skipping Dataset Success heatmaps."
        )
        return

    # Get unique algorithms and dataset types from the filtered data
    try:
        # --- MODIFICATION START ---
        # Filter out 'zeroshot' before getting unique algorithms for this specific plot type
        df_filtered_non_zeroshot = df_filtered[
            df_filtered["retrieval_algorithm"] != "zeroshot"
        ]
        if df_filtered_non_zeroshot.empty:
            print(
                f"No non-zeroshot data found for language '{target_language}' and metric '{target_metric}'. Skipping these heatmaps."
            )
            return

        unique_algos = df_filtered_non_zeroshot["retrieval_algorithm"].unique().tolist()
        # --- MODIFICATION END ---

        # Get unique datasets from the original filtered data (might include zeroshot datasets if needed elsewhere later)
        unique_datasets = df_filtered["dataset_type"].unique().tolist()
    except KeyError as e:
        print(
            f"Error: Missing expected column '{e}' in filtered data. Skipping Dataset Success heatmaps."
        )
        return

    if not unique_algos:
        print("No unique non-zeroshot retrieval algorithms found in the filtered data.")
        return
    if not unique_datasets:
        print("No unique dataset types found in the filtered data.")
        return

    # --- MODIFICATION: Updated print statement ---
    print(
        f"Found {len(unique_algos)} non-zeroshot algorithms and {len(unique_datasets)} dataset types for English Dataset Success heatmaps."
    )
    print(f"  Algorithms (excluding zeroshot): {unique_algos}")
    # --- End Modification ---
    print(f"  Dataset Types: {unique_datasets}")

    total_plots = len(unique_algos) * len(unique_datasets)
    plot_counter = 0

    # Iterate through each non-zeroshot algorithm and dataset type
    for algo in unique_algos:  # This loop now only contains non-zeroshot algos
        for dataset in unique_datasets:
            plot_counter += 1
            print(
                f"\n[{plot_counter}/{total_plots}] Generating Dataset Success Heatmap for: Lang={target_language}, Algo={algo}, Dataset={dataset}"
            )

            # Filter further for the specific algorithm and dataset type
            # Use df_filtered_non_zeroshot here as well to ensure consistency
            df_combo = df_filtered_non_zeroshot[
                (df_filtered_non_zeroshot["retrieval_algorithm"] == algo)
                & (df_filtered_non_zeroshot["dataset_type"] == dataset)
            ].copy()

            # Check if data exists and has variation for this specific combination
            # Need to check for NaN/None in chunk/overlap as well, although filtering zeroshot should handle it
            if (
                df_combo.empty
                or df_combo["chunk_size"].isnull().all()
                or df_combo["overlap_size"].isnull().all()
                or df_combo["chunk_size"].nunique() < 1
                or df_combo["overlap_size"].nunique() < 1
                or df_combo["question_model"].nunique() < 1
            ):
                print(
                    "  Skipping: Not enough data or variation (models, chunk/overlap) for this specific combination."
                )
                continue

            print(f"  Data points for this combination: {len(df_combo)}")

            # Prepare parameters for the plotting function
            fixed_params_combo = {
                "language": target_language,
                "retrieval_algorithm": algo,
                "dataset_type": dataset,
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
                values_col="metric_value",  # The actual data column name
                value_label="Success Rate",  # Specific label for this metric
                index_col_chunk="chunk_size",
                index_col_overlap="overlap_size",
                columns_col="question_model",
                sort_columns_by_value=True,  # Keep sorting models by performance
            )


def generate_algo_vs_model_f1_heatmap(
    df_data: pd.DataFrame,  # Takes the full dataframe
    output_dir: str,
    output_filename_prefix: str,
):
    """
    Generates a single heatmap comparing Retrieval Algorithms vs LLM Models based on F1 Score,
    specifically for the 'english' language.
    Averages F1 scores across all other parameters (chunk, overlap, etc.) within English results.
    """
    # --- MODIFICATION START ---
    target_language = "english"
    print(
        f"\nGenerating Heatmap: F1 Score - Algorithm vs Model (Language: {target_language})"
    )
    metric_to_plot = "f1_score"
    # Add 'language' to required columns
    required_cols = [
        "retrieval_algorithm",
        "question_model",
        "metric_type",
        "metric_value",
        "language",
    ]
    # --- MODIFICATION END ---

    if not all(col in df_data.columns for col in required_cols):
        print(
            f"Warning: Input DataFrame is missing one or more required columns for Algo vs Model F1 ({target_language}) heatmap: {required_cols}. Skipping."
        )
        return

    # 1. Filter for the relevant metric AND language
    df_filtered = df_data[
        (df_data["metric_type"] == metric_to_plot)
        & (df_data["language"] == target_language)
    ].copy()

    if df_filtered.empty:
        # --- MODIFICATION: Updated message ---
        print(
            f"No data found for metric_type '{metric_to_plot}' and language '{target_language}'. Skipping Algo vs Model F1 heatmap."
        )
        # --- MODIFICATION END ---
        return

    # 2. Aggregate data: Mean F1 score per Algo/Model combination (for English)
    try:
        df_agg = (
            df_filtered.groupby(["retrieval_algorithm", "question_model"])[
                "metric_value"
            ]
            .mean()
            .reset_index()
        )
        print(f"  Aggregated F1 data points ({target_language} only): {len(df_agg)}")
        if df_agg.empty:
            print("  Aggregation resulted in empty DataFrame. Skipping plot.")
            return
    except KeyError as e:
        # --- MODIFICATION: Updated message ---
        print(
            f"Error during aggregation for Algo vs Model F1 ({target_language}) heatmap: Missing column {e}. Skipping."
        )
        # --- MODIFICATION END ---
        return
    except Exception as e:
        # --- MODIFICATION: Updated message ---
        print(
            f"Error during aggregation for Algo vs Model F1 ({target_language}) heatmap: {e}. Skipping."
        )
        # --- MODIFICATION END ---
        return

    # 3. Prepare for plotting
    # --- MODIFICATION: Add language to filename ---
    output_filename = (
        f"{output_filename_prefix}f1_heatmap_algo_vs_model_{target_language}.png"
    )
    # --- MODIFICATION END ---
    output_filepath = os.path.join(output_dir, output_filename)

    # 4. Call the generic heatmap function
    create_f1_heatmap(  # Re-using the existing function
        data=df_agg,
        output_path=output_filepath,
        index_col="retrieval_algorithm",  # y-axis
        columns_col="question_model",  # x-axis
        values_col="metric_value",  # The aggregated mean F1 score
        value_label="Mean F1 Score",  # Label for the title/colorbar
        all_indices=None,  # Let it use indices present in aggregated data
        current_params=None,  # No specific sub-parameters for this plot
        # --- MODIFICATION: Update title ---
        title=f"Mean F1 Score (English): Retrieval Algorithm vs LLM Model",  # Custom title
        # --- MODIFICATION END ---
        sort_columns_by_value=True,  # Sort models by performance
    )


# --- MODIFIED: Generator for Algo vs Model Mean Dataset Success Heatmap (English Only) ---
def generate_algo_vs_model_dataset_success_heatmap(
    df_data: pd.DataFrame,  # Takes the full dataframe
    output_dir: str,
    output_filename_prefix: str,
):
    """
    Generates ONE heatmap for EACH specified dataset type, combining results
    across specified languages (English, German, French).
    Compares Algorithms/Languages (Rows) vs LLM Models (Columns) based on
    Dataset Success Rate. Rows are sorted by Algorithm/Noise Level first,
    then by Language (e.g., english_embedding, french_embedding...).
    Filters RAG algorithms by specific chunk/overlap and ZeroShot by specific noise levels.
    """
    # Define target languages and parameters
    languages_to_include = [
        "english",
        "german",
        "french",
    ]  # TARGET LANGUAGES TO INCLUDE
    metric_to_plot = "dataset_success"
    # Define target datasets based on config keys (adjust if keys differ)
    target_datasets = ["general_questions", "table_questions", "unanswerable_questions"]
    target_chunk = 200
    target_overlap = 100
    target_noise_levels = [1000, 10000, 30000, 59000]
    rag_algorithms_ordered = ["hybrid", "embedding", "keyword"]  # Desired RAG order
    # Create ordered list for zeroshot identifiers
    zeroshot_identifiers_ordered = [
        f"zeroshot_noise_{n}" for n in sorted(target_noise_levels)
    ]
    # Combine into the final algorithm order for sorting
    algorithm_sort_order = rag_algorithms_ordered + zeroshot_identifiers_ordered
    language_sort_order = languages_to_include  # Use the include list for order

    print(
        f"\nGenerating Dataset Success Heatmaps: Algorithm/Language vs Model (One per Dataset, Custom Sort)"
    )
    print(
        f"  Including Languages: {languages_to_include} (Order: {language_sort_order})"
    )
    print(f"  Target Metric: {metric_to_plot}")
    print(f"  Target Datasets: {target_datasets}")
    print(f"  Algorithm Sort Order: {algorithm_sort_order}")
    print(f"  RAG Params (fixed): C={target_chunk}/O={target_overlap}")
    print(f"  ZeroShot Noise Levels (included): {target_noise_levels}")

    required_cols = [
        "retrieval_algorithm",
        "question_model",
        "metric_type",
        "metric_value",
        "dataset_type",
        "language",
        "chunk_size",
        "overlap_size",
        "noise_level",
    ]
    # Check if all required columns are present
    missing_cols = [col for col in required_cols if col not in df_data.columns]
    if missing_cols:
        print(
            f"Warning: Input DataFrame is missing one or more required columns: {missing_cols}. Skipping these heatmaps."
        )
        return

    # Convert relevant columns to numeric ONCE for the whole dataframe, coercing errors
    df_data_processed = df_data.copy()  # Work on a copy
    numeric_cols = ["chunk_size", "overlap_size", "noise_level"]
    for col in numeric_cols:
        if col in df_data_processed.columns:
            df_data_processed[col] = pd.to_numeric(
                df_data_processed[col], errors="coerce"
            )

    # --- Filter ONCE for relevant languages and metric type ---
    df_metric_filtered = df_data_processed[
        (df_data_processed["language"].isin(languages_to_include))
        & (df_data_processed["metric_type"] == metric_to_plot)
    ].copy()  # Use the processed df with numeric types

    if df_metric_filtered.empty:
        print(
            f"  No data found for metric '{metric_to_plot}' and included languages '{languages_to_include}'. Skipping generation."
        )
        return  # Skip if no relevant data exists at all

    # --- Loop through dataset types ---
    dataset_plot_counter = 0
    for dataset_type in target_datasets:
        dataset_plot_counter += 1
        print(
            f"\n[{dataset_plot_counter}/{len(target_datasets)}] Processing dataset: {dataset_type} (All Languages)"
        )

        # Filter for the current dataset
        df_dataset_filtered = df_metric_filtered[
            df_metric_filtered["dataset_type"] == dataset_type
        ].copy()

        if df_dataset_filtered.empty:
            print(
                f"    No data found for dataset '{dataset_type}' (Metric: {metric_to_plot}, Langs: {languages_to_include}). Skipping heatmap."
            )
            continue

        # --- Filter RAG Data ---
        is_rag = df_dataset_filtered["retrieval_algorithm"] != "zeroshot"
        df_rag_filtered = df_dataset_filtered[
            is_rag
            & (df_dataset_filtered["chunk_size"] == target_chunk)
            & (df_dataset_filtered["overlap_size"] == target_overlap)
        ].copy()

        # Add combined plot index for RAG ('language_algorithm')
        if not df_rag_filtered.empty:
            # Combine language and algorithm for the plot index
            df_rag_filtered["plot_index"] = (
                df_rag_filtered["language"]
                + "_"
                + df_rag_filtered["retrieval_algorithm"]
            )
            df_rag_filtered["_sort_key_algo"] = df_rag_filtered[
                "retrieval_algorithm"
            ]  # Base algo is the sort key
            df_rag_filtered["_sort_key_lang"] = df_rag_filtered["language"]
            print(f"    Found {len(df_rag_filtered)} RAG points for {dataset_type}.")
        else:
            print(
                f"    No RAG data points found matching criteria for {dataset_type} across languages."
            )

        # --- Filter ZeroShot Data ---
        df_zeroshot_filtered = df_dataset_filtered[
            (df_dataset_filtered["retrieval_algorithm"] == "zeroshot")
            & (df_dataset_filtered["noise_level"].isin(target_noise_levels))
        ].copy()

        # Add combined plot index and sorting keys for ZeroShot
        if not df_zeroshot_filtered.empty:
            df_zeroshot_filtered["noise_level_int"] = (
                df_zeroshot_filtered["noise_level"].fillna(0).astype(int)
            )
            # Base identifier for sorting (e.g., 'zeroshot_noise_1000')
            df_zeroshot_filtered["_sort_key_algo"] = (
                "zeroshot_noise_" + df_zeroshot_filtered["noise_level_int"].astype(str)
            )
            # Combined index for display
            df_zeroshot_filtered["plot_index"] = (
                df_zeroshot_filtered["language"]
                + "_"
                + df_zeroshot_filtered["_sort_key_algo"]
            )
            df_zeroshot_filtered["_sort_key_lang"] = df_zeroshot_filtered["language"]

            df_zeroshot_filtered = df_zeroshot_filtered.drop(
                columns=["noise_level_int"]
            )
            print(
                f"    Found {len(df_zeroshot_filtered)} ZeroShot points for {dataset_type}."
            )
        else:
            print(
                f"    No ZeroShot data points found matching criteria for {dataset_type} across languages."
            )

        # --- Combine and Prepare for Plotting ---
        dfs_to_concat = [
            df for df in [df_rag_filtered, df_zeroshot_filtered] if not df.empty
        ]

        if not dfs_to_concat:
            print(
                f"    No data remaining for dataset '{dataset_type}' after applying RAG/ZeroShot filters across languages. Skipping heatmap."
            )
            continue

        df_combined = pd.concat(dfs_to_concat, ignore_index=True)

        if not all(
            col in df_combined.columns
            for col in [
                "plot_index",
                "question_model",
                "metric_value",
                "_sort_key_algo",
                "_sort_key_lang",
            ]
        ):
            print(
                f"    Error: Missing essential columns after concat for {dataset_type}. Skipping heatmap."
            )
            continue

        print(f"    Total points for heatmap ({dataset_type}): {len(df_combined)}")

        # --- Apply Custom Sorting ---
        try:
            # Convert sorting columns to Categorical with the defined order
            df_combined["_sort_key_algo"] = pd.Categorical(
                df_combined["_sort_key_algo"],
                categories=algorithm_sort_order,
                ordered=True,
            )
            df_combined["_sort_key_lang"] = pd.Categorical(
                df_combined["_sort_key_lang"],
                categories=language_sort_order,
                ordered=True,
            )

            # Sort the DataFrame
            df_combined_sorted = df_combined.sort_values(
                by=["_sort_key_algo", "_sort_key_lang"]
            )

            # Get the unique plot indices IN THE SORTED ORDER
            sorted_plot_indices = df_combined_sorted["plot_index"].unique().tolist()

            print(
                f"    Applied custom sort. Sorted indices (examples): {sorted_plot_indices[:10]}..."
            )
            print(
                f"    Unique models found: {df_combined_sorted['question_model'].unique().tolist()}"
            )

            # Remove temporary columns before plotting if desired
            df_plot_data = df_combined_sorted.drop(
                columns=["_sort_key_algo", "_sort_key_lang"]
            )

        except Exception as e:
            print(
                f"    Error during custom sorting for {dataset_type}: {e}. Falling back to default sort."
            )
            # Fallback: sort alphabetically by plot_index if custom sort fails
            df_combined_sorted = df_combined.sort_values(by="plot_index")
            sorted_plot_indices = df_combined_sorted["plot_index"].unique().tolist()
            df_plot_data = df_combined_sorted.drop(
                columns=["_sort_key_algo", "_sort_key_lang"], errors="ignore"
            )

        # --- Plotting ---
        sanitized_dataset_type = sanitize_filename(dataset_type)
        # Update filename to reflect multi-language nature (remove specific language)
        output_filename = f"{output_filename_prefix}dataset_success_heatmap_lang_algo_vs_model_{sanitized_dataset_type}.png"
        output_filepath = os.path.join(output_dir, output_filename)
        # Update title to reflect multi-language nature
        plot_title = (
            f"{dataset_type.replace('_', ' ').title()} Success Rate (Multi-Language, Sorted by Algo):\n"
            f"Algorithm/Language vs LLM Model (RAG: C={target_chunk}/O={target_overlap}, ZeroShot: Noise)"
        )

        # Call the generic heatmap function
        create_f1_heatmap(
            data=df_plot_data,  # Pass the data (sorting is handled by all_indices)
            output_path=output_filepath,
            index_col="plot_index",  # Use the combined index (Y-axis)
            columns_col="question_model",  # X-axis: Models
            values_col="metric_value",  # Cell values: Success Rate
            value_label="Success Rate",  # Label for the title/colorbar
            all_indices=sorted_plot_indices,  # <<< Pass the explicitly sorted index list
            current_params=None,
            title=plot_title,
            sort_columns_by_value=True,  # Sort models by performance (X-axis)
            # Adjust figsize if needed for potentially long Y-axis
            # figsize=(14, max(8, len(sorted_plot_indices) * 0.4)) # Dynamically adjust height?
        )


def generate_multilang_f1_score_report_heatmap(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str,
):
    """
    Generates a single F1 score heatmap combining results across specified languages.
    Compares Algorithms/Languages (Rows) vs LLM Models (Columns) based on overall F1 Score.
    Rows are sorted by Algorithm/Noise Level first, then by Language.
    Filters RAG algorithms by specific chunk/overlap and ZeroShot by specific noise levels.
    This plot is NOT dataset-bound; it uses the overall F1 score.
    """
    # Define target languages and parameters
    languages_to_include = [
        "english",
        "german",
        "french",
    ]  # TARGET LANGUAGES TO INCLUDE
    metric_to_plot = "f1_score"  # TARGET METRIC
    target_chunk = 200
    target_overlap = 100
    target_noise_levels = [1000, 10000, 30000, 59000]
    rag_algorithms_ordered = ["hybrid", "embedding", "keyword"]  # Desired RAG order
    # Create ordered list for zeroshot identifiers
    zeroshot_identifiers_ordered = [
        f"zeroshot_noise_{n}" for n in sorted(target_noise_levels)
    ]
    # Combine into the final algorithm order for sorting
    algorithm_sort_order = rag_algorithms_ordered + zeroshot_identifiers_ordered
    language_sort_order = languages_to_include  # Use the include list for order

    print(
        f"\nGenerating F1 Score Heatmap: Algorithm/Language vs Model (Multi-Language, Custom Sort)"
    )
    print(
        f"  Including Languages: {languages_to_include} (Order: {language_sort_order})"
    )
    print(f"  Target Metric: {metric_to_plot}")
    print(f"  Algorithm Sort Order: {algorithm_sort_order}")
    print(f"  RAG Params (fixed): C={target_chunk}/O={target_overlap}")
    print(f"  ZeroShot Noise Levels (included): {target_noise_levels}")

    required_cols = [
        "retrieval_algorithm",
        "question_model",
        "metric_type",
        "metric_value",
        "language",
        "chunk_size",
        "overlap_size",
        "noise_level",
        # "dataset_type" is NOT strictly required here as f1_score has dataset_type=None,
        # but it's good to have it in the df_data_processed for consistency if it comes from extract_detailed_visualization_data
    ]
    # Check if all required columns are present
    missing_cols = [col for col in required_cols if col not in df_data.columns]
    if missing_cols:
        print(
            f"Warning: Input DataFrame is missing one or more required columns for F1 Score heatmap: {missing_cols}. Skipping this heatmap."
        )
        return

    # Convert relevant columns to numeric ONCE for the whole dataframe, coercing errors
    df_data_processed = df_data.copy()  # Work on a copy
    numeric_cols = ["chunk_size", "overlap_size", "noise_level"]
    for col in numeric_cols:
        if col in df_data_processed.columns:
            df_data_processed[col] = pd.to_numeric(
                df_data_processed[col], errors="coerce"
            )

    # --- Filter ONCE for relevant languages and metric type ---
    df_metric_filtered = df_data_processed[
        (df_data_processed["language"].isin(languages_to_include))
        & (df_data_processed["metric_type"] == metric_to_plot)
    ].copy()

    if df_metric_filtered.empty:
        print(
            f"  No data found for metric '{metric_to_plot}' and included languages '{languages_to_include}'. Skipping F1 Score heatmap."
        )
        return

    # --- Filter RAG Data ---
    is_rag = df_metric_filtered["retrieval_algorithm"] != "zeroshot"
    df_rag_filtered = df_metric_filtered[
        is_rag
        & (df_metric_filtered["chunk_size"] == target_chunk)
        & (df_metric_filtered["overlap_size"] == target_overlap)
    ].copy()

    # Add combined plot index for RAG ('language_algorithm')
    if not df_rag_filtered.empty:
        df_rag_filtered["plot_index"] = (
            df_rag_filtered["language"] + "_" + df_rag_filtered["retrieval_algorithm"]
        )
        df_rag_filtered["_sort_key_algo"] = df_rag_filtered["retrieval_algorithm"]
        df_rag_filtered["_sort_key_lang"] = df_rag_filtered["language"]
        print(f"    Found {len(df_rag_filtered)} RAG F1 score points.")
    else:
        print(
            f"    No RAG F1 score data points found matching criteria across languages."
        )

    # --- Filter ZeroShot Data ---
    df_zeroshot_filtered = df_metric_filtered[
        (df_metric_filtered["retrieval_algorithm"] == "zeroshot")
        & (df_metric_filtered["noise_level"].isin(target_noise_levels))
    ].copy()

    # Add combined plot index and sorting keys for ZeroShot
    if not df_zeroshot_filtered.empty:
        df_zeroshot_filtered["noise_level_int"] = (
            df_zeroshot_filtered["noise_level"].fillna(0).astype(int)
        )
        df_zeroshot_filtered["_sort_key_algo"] = (
            "zeroshot_noise_" + df_zeroshot_filtered["noise_level_int"].astype(str)
        )
        df_zeroshot_filtered["plot_index"] = (
            df_zeroshot_filtered["language"]
            + "_"
            + df_zeroshot_filtered["_sort_key_algo"]
        )
        df_zeroshot_filtered["_sort_key_lang"] = df_zeroshot_filtered["language"]
        df_zeroshot_filtered = df_zeroshot_filtered.drop(columns=["noise_level_int"])
        print(f"    Found {len(df_zeroshot_filtered)} ZeroShot F1 score points.")
    else:
        print(
            f"    No ZeroShot F1 score data points found matching criteria across languages."
        )

    # --- Combine and Prepare for Plotting ---
    dfs_to_concat = [
        df for df in [df_rag_filtered, df_zeroshot_filtered] if not df.empty
    ]

    if not dfs_to_concat:
        print(
            f"    No F1 score data remaining after applying RAG/ZeroShot filters across languages. Skipping heatmap."
        )
        return

    df_combined = pd.concat(dfs_to_concat, ignore_index=True)

    if not all(
        col in df_combined.columns
        for col in [
            "plot_index",
            "question_model",
            "metric_value",
            "_sort_key_algo",
            "_sort_key_lang",
        ]
    ):
        print(
            f"    Error: Missing essential columns after concat for F1 score heatmap. Skipping heatmap."
        )
        return

    print(f"    Total F1 score points for heatmap: {len(df_combined)}")

    # --- Apply Custom Sorting ---
    try:
        df_combined["_sort_key_algo"] = pd.Categorical(
            df_combined["_sort_key_algo"], categories=algorithm_sort_order, ordered=True
        )
        df_combined["_sort_key_lang"] = pd.Categorical(
            df_combined["_sort_key_lang"], categories=language_sort_order, ordered=True
        )
        df_combined_sorted = df_combined.sort_values(
            by=["_sort_key_algo", "_sort_key_lang"]
        )
        sorted_plot_indices = df_combined_sorted["plot_index"].unique().tolist()
        print(
            f"    Applied custom sort for F1 score heatmap. Sorted indices (examples): {sorted_plot_indices[:10]}..."
        )
        df_plot_data = df_combined_sorted.drop(
            columns=["_sort_key_algo", "_sort_key_lang"]
        )
    except Exception as e:
        print(
            f"    Error during custom sorting for F1 score heatmap: {e}. Falling back to default sort."
        )
        df_combined_sorted = df_combined.sort_values(by="plot_index")
        sorted_plot_indices = df_combined_sorted["plot_index"].unique().tolist()
        df_plot_data = df_combined_sorted.drop(
            columns=["_sort_key_algo", "_sort_key_lang"], errors="ignore"
        )

    # --- Plotting ---
    output_filename = (
        f"{output_filename_prefix}f1_score_heatmap_multilang_algo_vs_model.png"
    )
    output_filepath = os.path.join(output_dir, output_filename)
    plot_title = (
        f"F1 Score (Multi-Language, Sorted by Algo):\n"
        f"Algorithm/Language vs LLM Model (RAG: C={target_chunk}/O={target_overlap}, ZeroShot: Noise)"
    )

    create_f1_heatmap(
        data=df_plot_data,
        output_path=output_filepath,
        index_col="plot_index",
        columns_col="question_model",
        values_col="metric_value",
        value_label="F1 Score",  # Specific label for F1
        all_indices=sorted_plot_indices,
        current_params=None,
        title=plot_title,
        sort_columns_by_value=True,
        # figsize=(14, max(8, len(sorted_plot_indices) * 0.4)) # Optional dynamic figsize
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

    parser = argparse.ArgumentParser(
        description="Generate specific F1 heatmap visualizations."
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
        "--config-path",  # Needed for language list
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
            "dataset_success",  # This is the Model vs Chunk/Overlap for Dataset Success
            "algo_vs_model_f1",  # New choice for F1 Algo vs Model
            "algo_vs_model_success",  # New choice for Mean Success Algo vs Model (now per dataset)
            "multilang_f1_report",  # New choice for the F1 score multi-language report
            "all",
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
            loaded_languages = [
                lc.get("language") for lc in language_configs if lc.get("language")
            ]
            if loaded_languages:
                all_languages_list = loaded_languages
                print(f"Loaded languages for heatmap axes/order: {all_languages_list}")
            else:
                print(
                    f"Warning: No languages found in {args.config_path}. Heatmap axes order may vary."
                )
        except FileNotFoundError:
            print(
                f"Warning: Config file not found at '{args.config_path}'. Cannot determine language list."
            )
        except Exception as e:
            print(f"Warning: Error loading config file '{args.config_path}': {e}.")

    # 2. Generate the requested plot subtype(s)
    subtypes_to_generate = []
    if args.plot_subtype == "all":
        subtypes_to_generate = [
            "lang_vs_model",
            "chunk_vs_overlap",
            "model_vs_chunk_overlap",
            "dataset_success",
            "algo_vs_model_f1",
            "algo_vs_model_success",
            "multilang_f1_report", # Add new type to 'all'
        ]
    else:
        subtypes_to_generate = [args.plot_subtype]

    for subtype in subtypes_to_generate:
        if subtype == "lang_vs_model":
            # This one specifically needs F1 data
            df_f1_heatmap = df_data[df_data["metric_type"] == "f1_score"].copy()
            if df_f1_heatmap.empty:
                print(
                    "Warning: No F1 score data found. Skipping lang_vs_model heatmap."
                )
                continue
            generate_language_vs_model_heatmap(
                df_f1_heatmap=df_f1_heatmap,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
                all_languages_list=all_languages_list,
            )
        elif subtype == "chunk_vs_overlap":
            # This one specifically needs F1 data
            df_f1_heatmap = df_data[df_data["metric_type"] == "f1_score"].copy()
            if df_f1_heatmap.empty:
                print(
                    "Warning: No F1 score data found. Skipping chunk_vs_overlap heatmap."
                )
                continue
            generate_chunk_vs_overlap_heatmap(
                df_f1_heatmap=df_f1_heatmap,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
            )
        elif subtype == "model_vs_chunk_overlap":
            # This one specifically needs F1 data
            df_f1_heatmap = df_data[df_data["metric_type"] == "f1_score"].copy()
            if df_f1_heatmap.empty:
                print(
                    "Warning: No F1 score data found. Skipping model_vs_chunk_overlap heatmap."
                )
                continue
            generate_model_vs_chunk_overlap_heatmap(
                df_f1_heatmap=df_f1_heatmap,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
            )
        elif (
            subtype == "dataset_success"
        ):  # This is the Model vs Chunk/Overlap for Dataset Success
            generate_dataset_success_heatmaps(
                df_data=df_data,  # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
            )
        # --- NEW: Call the new generator functions ---
        # mean F1 heatmap for algo vs model (english only)
        # elif subtype == "algo_vs_model_f1":
        #     generate_algo_vs_model_f1_heatmap(
        #         df_data=df_data,  # Pass the full dataframe
        #         output_dir=args.output_dir,
        #         output_filename_prefix=args.output_filename_prefix,
        #     )
        elif subtype == "algo_vs_model_success":
            generate_algo_vs_model_dataset_success_heatmap(
                df_data=df_data,  # Pass the full dataframe
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
            )
        elif subtype == "multilang_f1_report": # Call the new F1 report generator
            generate_multilang_f1_score_report_heatmap(
                df_data=df_data,
                output_dir=args.output_dir,
                output_filename_prefix=args.output_filename_prefix,
            )

    print("\n--- Standalone Heatmap Generation Finished ---")
