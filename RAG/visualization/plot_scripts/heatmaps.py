# visualization/plot_scripts/heatmaps.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Optional, Tuple, List, Dict, Any # Added Dict, Any

# --- DEBUG FLAG ---
# Set to True to enable detailed printing, False to disable
DEBUG_HEATMAP = True

def create_f1_heatmap(
    data: pd.DataFrame,
    output_path: str,
    index_col: str = 'language',
    columns_col: str = 'question_model',
    values_col: str = 'f1_score',
    all_indices: Optional[List[str]] = None, # List of all expected index values (e.g., all languages)
    current_params: Optional[Dict[str, Any]] = None, # Specific parameters for this heatmap (for title)
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    annot_fmt: str = ".3f",
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Creates a heatmap of F1 scores, typically with languages as rows and models as columns,
    for a specific set of RAG parameters (passed via current_params for the title).
    Ensures all specified indices (e.g., languages) are included, even if data is missing.

    Args:
        data: Pandas DataFrame containing the ALREADY FILTERED results data for a specific
              parameter combination. Must include columns specified by index_col,
              columns_col, and values_col.
        output_path: The full path where the plot image will be saved.
        index_col: The column name to use for the heatmap rows (index). Default: 'language'.
        columns_col: The column name to use for the heatmap columns. Default: 'question_model'.
        values_col: The column name containing the scores for the heatmap cells. Default: 'f1_score'.
        all_indices: Optional list of all expected values for the index (e.g., ['english', 'french', 'german']).
                     If provided, the heatmap index will be forced to include all these values,
                     filling missing data with NaN. Default: None (uses only indices present in data).
        current_params: Optional dictionary containing the specific parameters (e.g.,
                        {'retrieval_algorithm': 'embedding', 'chunk_size': 2000, 'overlap_size': 100})
                        used to generate this heatmap's data, primarily for the title.
        title: Optional title for the plot. If None, a default title is generated using current_params.
        xlabel: Optional label for the x-axis (columns). If None, uses columns_col.
        ylabel: Optional label for the y-axis (rows). If None, uses index_col.
        cmap: Colormap to use for the heatmap (e.g., "viridis", "coolwarm", "YlGnBu").
        annot_fmt: String format for annotations within the heatmap cells. Default: ".3f".
        figsize: Tuple specifying the figure size (width, height) in inches.
    """
    if data is None or data.empty:
        # This case should ideally be handled by the calling script before calling this function
        print(f"Warning: Input data for heatmap '{output_path}' is None or empty. Skipping plot generation.")
        return
    if not all(col in data.columns for col in [index_col, columns_col, values_col]):
        print(f"Error: DataFrame must contain columns: '{index_col}', '{columns_col}', '{values_col}'.")
        print(f"Available columns: {data.columns.tolist()}")
        return

    if DEBUG_HEATMAP:
        print("\n--- Heatmap Debug: Received Data (Should be pre-filtered) ---")
        # Print relevant columns for easier checking
        cols_to_print = [index_col, columns_col, values_col]
        if current_params:
             cols_to_print.extend(current_params.keys())
        cols_to_print = list(set(col for col in cols_to_print if col in data.columns)) # Ensure columns exist
        print(data[cols_to_print].to_string())
        print("-" * 30)

    # --- Pivot Data ---
    # Data is assumed to be pre-filtered, so we pivot directly
    try:
        if DEBUG_HEATMAP: print("--- Heatmap Debug: Creating pivot table ---")
        pivot_df = pd.pivot_table(
            data, # Use the passed (already filtered) data
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc='mean' # Use mean if multiple runs exist for the exact same lang/model/params
        )
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Pivot table BEFORE reindexing ---\n{pivot_df.to_string()}\n" + "-"*30)
    except Exception as e:
        print(f"Error creating pivot table for {output_path}: {e}")
        return

    # --- Ensure all expected indices are present ---
    if all_indices:
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Ensuring heatmap index includes: {all_indices} ---")
        # Ensure index name matches before reindexing if it exists
        if pivot_df.index.name != index_col:
             pivot_df.index.name = index_col
        pivot_df = pivot_df.reindex(all_indices)
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Pivot table AFTER reindexing ---\n{pivot_df.to_string()}\n" + "-"*30)

    if pivot_df.isnull().all().all():
         print(f"Warning: Pivot table for {output_path} contains only NaN values after pivoting and reindexing. Heatmap might be blank.")
         # Optionally, you could skip saving a completely blank heatmap
         # return

    # --- Plotting ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="white")
    plt.figure(figsize=figsize)

    if DEBUG_HEATMAP: print("--- Heatmap Debug: Generating heatmap plot ---")
    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        linewidths=.5,
        linecolor='lightgray',
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        # Handle NaN display explicitly if needed, though default usually works
        # You might need `annot_kws={"size": 8}` if text overlaps
    )

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Construct title based on passed parameters
    default_title = f'{values_col.replace("_", " ").title()} ({index_col.title()} vs {columns_col.title()})'
    if current_params:
        # Format the parameter string more nicely
        param_parts = []
        if 'retrieval_algorithm' in current_params:
            param_parts.append(f"Algo: {current_params['retrieval_algorithm']}")
        if 'chunk_size' in current_params:
            param_parts.append(f"Chunk: {current_params['chunk_size']}")
        if 'overlap_size' in current_params:
            param_parts.append(f"Overlap: {current_params['overlap_size']}")
        # Add other params if needed in the future
        param_str = ", ".join(param_parts)
        default_title += f'\n({param_str})'
    plot_title = title or default_title
    x_axis_label = xlabel or columns_col.replace("_", " ").title()
    y_axis_label = ylabel or index_col.replace("_", " ").title()

    plt.title(plot_title, fontsize=14, pad=20)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving heatmap to {output_path}: {e}")

    plt.close()

# Example usage part remains the same, but the call within main_visualization.py will change.
if __name__ == '__main__':
    # This part is for demonstration; assumes extract_visualization_data is available
    # and you have some results files.
    print("--- Testing Heatmap Creation (Example - Run main_visualization.py for actual use) ---")

    # Simulate data extraction (replace with actual call in main script)
    # Find the project root relative to this script
    current_script_dir = os.path.dirname(__file__)
    visualization_dir = os.path.dirname(current_script_dir)
    project_root_dir = os.path.dirname(visualization_dir)
    sys.path.insert(0, project_root_dir) # Add project root to path

    try:
        # Need ConfigLoader to get languages for the example
        from utils.config_loader import ConfigLoader
        from visualization.visualization_data_extractor import extract_visualization_data

        # Load config to get languages
        config_path = os.path.join(project_root_dir, 'config.json')
        if not os.path.exists(config_path):
             # Fallback if main config is missing, use a default list
             print(f"Warning: Config file not found at {config_path}. Using default language list for example.")
             all_languages_list = ['english', 'french', 'german'] # Default example
        else:
            config_loader = ConfigLoader(config_path)
            language_configs = config_loader.config.get("language_configs", [])
            all_languages_list = [lc.get("language") for lc in language_configs if lc.get("language")]
            if not all_languages_list:
                 print("Warning: No languages found in config file. Using default list for example.")
                 all_languages_list = ['english', 'french', 'german'] # Default example


        results_dir = os.path.join(project_root_dir, 'results')
        print(f"Attempting to load data from: {results_dir}")
        example_df = extract_visualization_data(results_dir)

        if example_df is not None and not example_df.empty:
            # Define output path for the test
            test_output_dir = os.path.join(visualization_dir, "plots_test") # Save to a test subdir

            # --- Simulate the filtering and calling process from main_visualization.py ---
            # 1. Define the specific parameters for this simulated heatmap
            test_params = {
                'retrieval_algorithm': 'embedding', # Adjust if needed
                'chunk_size': 2000,                # Adjust if needed
                'overlap_size': 100                 # Adjust if needed
            }
            print(f"\nSimulating heatmap generation for parameters: {test_params}")
            print(f"Using language list for reindexing: {all_languages_list}")

            # 2. Filter the DataFrame based on these parameters
            filtered_df_for_test = example_df[
                (example_df['retrieval_algorithm'] == test_params['retrieval_algorithm']) &
                (example_df['chunk_size'] == test_params['chunk_size']) &
                (example_df['overlap_size'] == test_params['overlap_size'])
            ]

            if not filtered_df_for_test.empty:
                # 3. Define the output path including parameters
                param_str_filename = f"algo_{test_params['retrieval_algorithm']}_cs_{test_params['chunk_size']}_os_{test_params['overlap_size']}"
                test_output_path = os.path.join(test_output_dir, f"test_f1_heatmap_{param_str_filename}.png")

                # 4. Call the heatmap function with the filtered data and parameters
                create_f1_heatmap(
                    data=filtered_df_for_test, # Pass the filtered data
                    output_path=test_output_path,
                    all_indices=all_languages_list, # Pass the language list
                    current_params=test_params,     # Pass the specific params for the title
                    # title="Example Heatmap (Filtered Data)" # Optional: override default title
                )
            else:
                print(f"No data found in '{results_dir}' matching the test parameters: {test_params}. Cannot generate example heatmap.")

        else:
            print("Could not load example data. Ensure 'results' directory exists and contains valid files.")
            print("Run 'main_visualization.py' for actual generation.")

    except ImportError as e:
        print(f"Could not import necessary modules (extract_visualization_data, ConfigLoader). Run 'main_visualization.py' for full functionality. Error: {e}")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Heatmap Creation Test Finished ---")