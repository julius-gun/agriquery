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
        print("\n--- Heatmap Debug (create_f1_heatmap): Received Data (Should be pre-filtered) ---")
        # Print relevant columns for easier checking
        cols_to_print = [index_col, columns_col, values_col]
        if current_params:
             # Add keys from current_params if they exist as columns in the data
             cols_to_print.extend([k for k in current_params.keys() if k in data.columns])
        cols_to_print = list(set(cols_to_print)) # Ensure unique columns
        print(data[cols_to_print].to_string())
        print("-" * 30)

    # --- Pivot Data ---
    # Data is assumed to be pre-filtered, so we pivot directly
    try:
        if DEBUG_HEATMAP: print("--- Heatmap Debug (create_f1_heatmap): Creating pivot table ---")
        pivot_df = pd.pivot_table(
            data, # Use the passed (already filtered) data
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc='mean' # Use mean if multiple runs exist for the exact same lang/model/params
        )
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug (create_f1_heatmap): Pivot table BEFORE reindexing ---\n{pivot_df.to_string()}\n" + "-"*30)
    except Exception as e:
        print(f"Error creating pivot table for {output_path}: {e}")
        return

    # --- Ensure all expected indices are present ---
    if all_indices:
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug (create_f1_heatmap): Ensuring heatmap index includes: {all_indices} ---")
        # Ensure index name matches before reindexing if it exists
        if pivot_df.index.name != index_col:
             pivot_df.index.name = index_col
        pivot_df = pivot_df.reindex(all_indices)
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug (create_f1_heatmap): Pivot table AFTER reindexing ---\n{pivot_df.to_string()}\n" + "-"*30)

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

    if DEBUG_HEATMAP: print("--- Heatmap Debug (create_f1_heatmap): Generating heatmap plot ---")
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
        # Use keys from current_params directly
        for key, value in current_params.items():
             param_parts.append(f"{key.replace('_', ' ').title()}: {value}")
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


# --- NEW FUNCTION ---
def create_chunk_overlap_heatmap(
    data: pd.DataFrame,
    output_path: str,
    values_col: str = 'f1_score',
    index_col: str = 'chunk_size',
    columns_col: str = 'overlap_size',
    fixed_params: Optional[Dict[str, Any]] = None, # Parameters held constant (for title)
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    annot_fmt: str = ".3f",
    figsize: Tuple[int, int] = (10, 7), # Adjusted default size
):
    """
    Creates a heatmap showing a metric (e.g., F1 score) against combinations of
    chunk size and overlap size, for a specific set of fixed parameters (e.g.,
    a specific language, model, and retrieval algorithm).

    Args:
        data: Pandas DataFrame containing the ALREADY FILTERED results data for the
              fixed parameters. Must include columns specified by index_col,
              columns_col, and values_col, corresponding to various chunk/overlap runs.
        output_path: The full path where the plot image will be saved.
        values_col: The column name containing the scores for the heatmap cells. Default: 'f1_score'.
        index_col: The column name to use for the heatmap rows (index). Default: 'chunk_size'.
        columns_col: The column name to use for the heatmap columns. Default: 'overlap_size'.
        fixed_params: Optional dictionary containing the parameters that were held constant
                      (e.g., {'language': 'english', 'question_model': 'model_x', 'retrieval_algorithm': 'algo_y'})
                      used to generate this heatmap's data, primarily for the title.
        title: Optional title for the plot. If None, a default title is generated using fixed_params.
        xlabel: Optional label for the x-axis (columns). If None, uses columns_col.
        ylabel: Optional label for the y-axis (rows). If None, uses index_col.
        cmap: Colormap to use for the heatmap (e.g., "viridis", "coolwarm", "YlGnBu").
        annot_fmt: String format for annotations within the heatmap cells. Default: ".3f".
        figsize: Tuple specifying the figure size (width, height) in inches.
    """
    if data is None or data.empty:
        print(f"Warning: Input data for chunk/overlap heatmap '{output_path}' is None or empty. Skipping plot generation.")
        return
    if not all(col in data.columns for col in [index_col, columns_col, values_col]):
        print(f"Error: DataFrame must contain columns: '{index_col}', '{columns_col}', '{values_col}'.")
        print(f"Available columns: {data.columns.tolist()}")
        return

    if DEBUG_HEATMAP:
        print("\n--- Heatmap Debug (create_chunk_overlap_heatmap): Received Data (Should be pre-filtered) ---")
        # Print relevant columns for easier checking
        cols_to_print = [index_col, columns_col, values_col]
        if fixed_params:
             # Add keys from fixed_params if they exist as columns in the data
             cols_to_print.extend([k for k in fixed_params.keys() if k in data.columns])
        cols_to_print = list(set(cols_to_print)) # Ensure unique columns
        print(data[cols_to_print].to_string())
        print("-" * 30)

    # --- Pivot Data ---
    try:
        if DEBUG_HEATMAP: print("--- Heatmap Debug (create_chunk_overlap_heatmap): Creating pivot table ---")
        pivot_df = pd.pivot_table(
            data, # Use the passed (already filtered) data
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc='mean' # Use mean if multiple runs exist for the exact same chunk/overlap/fixed_params
        )
        # Sort index/columns numerically for better readability
        pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug (create_chunk_overlap_heatmap): Pivot table ---\n{pivot_df.to_string()}\n" + "-"*30)
    except KeyError as e:
         print(f"Error creating pivot table for {output_path}: Missing column {e}. Ensure '{index_col}', '{columns_col}', and '{values_col}' exist in the filtered data.")
         return
    except Exception as e:
        print(f"Error creating pivot table for {output_path}: {e}")
        return

    if pivot_df.isnull().all().all():
         print(f"Warning: Pivot table for {output_path} contains only NaN values. Heatmap might be blank.")
         # return # Optionally skip saving blank heatmaps

    # --- Plotting ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="white")
    plt.figure(figsize=figsize)

    if DEBUG_HEATMAP: print("--- Heatmap Debug (create_chunk_overlap_heatmap): Generating heatmap plot ---")
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
    )

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Construct title based on fixed parameters
    default_title = f'{values_col.replace("_", " ").title()} ({index_col.replace("_", " ").title()} vs {columns_col.replace("_", " ").title()})'
    if fixed_params:
        param_parts = []
        # Define preferred order or filter specific keys if needed
        preferred_order = ['language', 'question_model', 'retrieval_algorithm'] # Example order
        sorted_keys = [k for k in preferred_order if k in fixed_params] + \
                      [k for k in fixed_params if k not in preferred_order]

        for key in sorted_keys:
            param_parts.append(f"{key.replace('_', ' ').title()}: {fixed_params[key]}")
        param_str = ", ".join(param_parts)
        default_title += f'\n(Fixed: {param_str})'
    plot_title = title or default_title
    x_axis_label = xlabel or columns_col.replace("_", " ").title()
    y_axis_label = ylabel or index_col.replace("_", " ").title()

    plt.title(plot_title, fontsize=14, pad=20)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chunk/Overlap Heatmap saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving chunk/overlap heatmap to {output_path}: {e}")

    plt.close()

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

[...] # Keep create_f1_heatmap as is

[...] # Keep create_chunk_overlap_heatmap as is


# --- NEW FUNCTION ---
def create_model_vs_chunk_overlap_heatmap(
    data: pd.DataFrame,
    output_path: str,
    values_col: str = 'f1_score',
    index_col_chunk: str = 'chunk_size',
    index_col_overlap: str = 'overlap_size',
    columns_col: str = 'question_model',
    fixed_params: Optional[Dict[str, Any]] = None, # Parameters held constant (e.g., language, algorithm)
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    annot_fmt: str = ".3f",
    figsize: Tuple[int, int] = (14, 10), # Adjusted default size
):
    """
    Creates a heatmap showing a metric (e.g., F1 score) against combinations of
    chunk size/overlap size (rows) and question models (columns), for a specific
    set of fixed parameters (e.g., a specific language and retrieval algorithm).

    Args:
        data: Pandas DataFrame containing the ALREADY FILTERED results data for the
              fixed parameters (e.g., language='english', retrieval_algorithm='embedding').
              Must include columns specified by index_col_chunk, index_col_overlap,
              columns_col, and values_col.
        output_path: The full path where the plot image will be saved.
        values_col: The column name containing the scores for the heatmap cells. Default: 'f1_score'.
        index_col_chunk: Column name for chunk size (part of the MultiIndex). Default: 'chunk_size'.
        index_col_overlap: Column name for overlap size (part of the MultiIndex). Default: 'overlap_size'.
        columns_col: The column name to use for the heatmap columns. Default: 'question_model'.
        fixed_params: Optional dictionary containing the parameters that were held constant
                      (e.g., {'language': 'english', 'retrieval_algorithm': 'algo_y'})
                      used to generate this heatmap's data, primarily for the title.
        title: Optional title for the plot. If None, a default title is generated using fixed_params.
        xlabel: Optional label for the x-axis (columns). If None, uses columns_col.
        ylabel: Optional label for the y-axis (rows). If None, uses "Chunk Size / Overlap Size".
        cmap: Colormap to use for the heatmap (e.g., "viridis", "coolwarm", "YlGnBu").
        annot_fmt: String format for annotations within the heatmap cells. Default: ".3f".
        figsize: Tuple specifying the figure size (width, height) in inches.
    """
    if data is None or data.empty:
        print(f"Warning: Input data for model vs chunk/overlap heatmap '{output_path}' is None or empty. Skipping plot generation.")
        return
    required_cols = [index_col_chunk, index_col_overlap, columns_col, values_col]
    if not all(col in data.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}.")
        print(f"Available columns: {data.columns.tolist()}")
        return

    if DEBUG_HEATMAP:
        print("\n--- Heatmap Debug (create_model_vs_chunk_overlap_heatmap): Received Data (Should be pre-filtered) ---")
        # Print relevant columns for easier checking
        cols_to_print = required_cols[:] # Copy the list
        if fixed_params:
             # Add keys from fixed_params if they exist as columns in the data
             cols_to_print.extend([k for k in fixed_params.keys() if k in data.columns])
        cols_to_print = list(set(cols_to_print)) # Ensure unique columns
        print(data[cols_to_print].to_string())
        print("-" * 30)

    # --- Pivot Data ---
    try:
        if DEBUG_HEATMAP: print("--- Heatmap Debug (create_model_vs_chunk_overlap_heatmap): Creating pivot table ---")
        pivot_df = pd.pivot_table(
            data, # Use the passed (already filtered) data
            values=values_col,
            index=[index_col_chunk, index_col_overlap], # Use MultiIndex for rows
            columns=columns_col,
            aggfunc='mean' # Use mean if multiple runs exist for the exact same combo
        )
        # Sort MultiIndex (numerically by chunk then overlap) and columns (alphabetically)
        pivot_df = pivot_df.sort_index(level=[0, 1], ascending=[True, True]).sort_index(axis=1)
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug (create_model_vs_chunk_overlap_heatmap): Pivot table ---\n{pivot_df.to_string()}\n" + "-"*30)
    except KeyError as e:
         print(f"Error creating pivot table for {output_path}: Missing column {e}. Ensure {required_cols} exist in the filtered data.")
         return
    except Exception as e:
        print(f"Error creating pivot table for {output_path}: {e}")
        return

    if pivot_df.isnull().all().all():
         print(f"Warning: Pivot table for {output_path} contains only NaN values. Heatmap might be blank.")
         # return # Optionally skip saving blank heatmaps

    # --- Plotting ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="white")
    plt.figure(figsize=figsize)

    if DEBUG_HEATMAP: print("--- Heatmap Debug (create_model_vs_chunk_overlap_heatmap): Generating heatmap plot ---")
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
        # Consider annot_kws={"size": 8} if annotations overlap
    )

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0) # MultiIndex labels might get long, but 0 rotation is usually fine

    # Construct title based on fixed parameters
    value_title = values_col.replace("_", " ").title()
    default_title = f'{value_title} (Model vs Chunk/Overlap)'
    if fixed_params:
        param_parts = []
        # Define preferred order or filter specific keys if needed
        preferred_order = ['language', 'retrieval_algorithm'] # Example order
        sorted_keys = [k for k in preferred_order if k in fixed_params] + \
                      [k for k in fixed_params if k not in preferred_order]

        for key in sorted_keys:
            param_parts.append(f"{key.replace('_', ' ').title()}: {fixed_params[key]}")
        param_str = ", ".join(param_parts)
        default_title += f'\n(Fixed: {param_str})'
    plot_title = title or default_title
    x_axis_label = xlabel or columns_col.replace("_", " ").title()
    # Custom Y label for MultiIndex
    y_axis_label = ylabel or f"{index_col_chunk.replace('_', ' ').title()} / {index_col_overlap.replace('_', ' ').title()}"

    plt.title(plot_title, fontsize=14, pad=20)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Model vs Chunk/Overlap Heatmap saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving model vs chunk/overlap heatmap to {output_path}: {e}")

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
            os.makedirs(test_output_dir, exist_ok=True) # Ensure test dir exists

            # --- Simulate the filtering and calling process for create_f1_heatmap ---
            print("\n--- Testing create_f1_heatmap ---")
            test_params_f1 = {
                'retrieval_algorithm': 'embedding', # Adjust if needed based on your results files
                'chunk_size': 2000,                # Adjust if needed
                'overlap_size': 100                 # Adjust if needed
            }
            print(f"Simulating heatmap generation for parameters: {test_params_f1}")
            print(f"Using language list for reindexing: {all_languages_list}")

            # 2. Filter the DataFrame based on these parameters
            filtered_df_for_f1_test = example_df[
                (example_df['retrieval_algorithm'] == test_params_f1['retrieval_algorithm']) &
                (example_df['chunk_size'] == test_params_f1['chunk_size']) &
                (example_df['overlap_size'] == test_params_f1['overlap_size'])
            ]

            if not filtered_df_for_f1_test.empty:
                # 3. Define the output path including parameters
                param_str_filename_f1 = f"algo_{test_params_f1['retrieval_algorithm']}_cs_{test_params_f1['chunk_size']}_os_{test_params_f1['overlap_size']}"
                test_output_path_f1 = os.path.join(test_output_dir, f"test_f1_heatmap_{param_str_filename_f1}.png")

                # 4. Call the heatmap function with the filtered data and parameters
                create_f1_heatmap(
                    data=filtered_df_for_f1_test, # Pass the filtered data
                    output_path=test_output_path_f1,
                    all_indices=all_languages_list, # Pass the language list
                    current_params=test_params_f1,     # Pass the specific params for the title
                    # title="Example Heatmap (Filtered Data)" # Optional: override default title
                )
            else:
                print(f"No data found for create_f1_heatmap test matching parameters: {test_params_f1}.")
                print("Check if your 'results' directory contains files matching these parameters.")

            # --- Simulate the filtering and calling process for create_chunk_overlap_heatmap ---
            print("\n--- Testing create_chunk_overlap_heatmap ---")
            # Define the fixed parameters for this specific heatmap
            fixed_params_chunk = {
                'language': 'english',             # Fixed language
                'question_model': 'llama3.2_1B-128k', # Fixed model (adjust to one present in your results)
                'retrieval_algorithm': 'embedding' # Fixed algorithm
            }
            print(f"Simulating chunk/overlap heatmap generation for fixed parameters: {fixed_params_chunk}")

            # Filter the main DataFrame for these fixed parameters
            # This subset should contain rows with varying chunk_size and overlap_size
            filtered_df_for_chunk_test = example_df[
                (example_df['language'] == fixed_params_chunk['language']) &
                (example_df['question_model'] == fixed_params_chunk['question_model']) &
                (example_df['retrieval_algorithm'] == fixed_params_chunk['retrieval_algorithm'])
            ]

            if not filtered_df_for_chunk_test.empty:
                 # Define the output path including fixed parameters
                 fixed_param_str_filename_chunk = f"lang_{fixed_params_chunk['language']}_model_{fixed_params_chunk['question_model'].replace(':','-')}_algo_{fixed_params_chunk['retrieval_algorithm']}"
                 test_output_path_chunk = os.path.join(test_output_dir, f"test_chunk_overlap_f1_heatmap_{fixed_param_str_filename_chunk}.png")

                 # Call the new heatmap function
                 create_chunk_overlap_heatmap(
                     data=filtered_df_for_chunk_test, # Pass the filtered data
                     output_path=test_output_path_chunk,
                     fixed_params=fixed_params_chunk, # Pass the fixed params for the title
                     values_col='f1_score',           # Explicitly state the value to plot
                     index_col='chunk_size',          # Rows = chunk size
                     columns_col='overlap_size'       # Columns = overlap size
                 )
            else:
                 print(f"No data found for create_chunk_overlap_heatmap test matching fixed parameters: {fixed_params_chunk}.")
                 print("Ensure your 'results' directory contains files matching these fixed parameters with varying chunk/overlap sizes.")

            # --- NEW: Simulate filtering and calling for create_model_vs_chunk_overlap_heatmap ---
            print("\n--- Testing create_model_vs_chunk_overlap_heatmap ---")
            # Define the fixed parameters for this heatmap (e.g., English language, specific algorithm)
            fixed_params_model_chunk = {
                'language': 'english',             # Fixed language (as requested)
                'retrieval_algorithm': 'embedding' # Fixed algorithm (adjust if needed)
            }
            print(f"Simulating model vs chunk/overlap heatmap generation for fixed parameters: {fixed_params_model_chunk}")

            # Filter the main DataFrame for these fixed parameters
            # This subset should contain rows with varying chunk_size, overlap_size, and question_model
            filtered_df_for_model_chunk_test = example_df[
                (example_df['language'] == fixed_params_model_chunk['language']) &
                (example_df['retrieval_algorithm'] == fixed_params_model_chunk['retrieval_algorithm'])
            ]

            if not filtered_df_for_model_chunk_test.empty:
                 # Define the output path including fixed parameters
                 fixed_param_str_filename_model_chunk = f"lang_{fixed_params_model_chunk['language']}_algo_{fixed_params_model_chunk['retrieval_algorithm']}"
                 test_output_path_model_chunk = os.path.join(test_output_dir, f"test_model_vs_chunk_overlap_f1_heatmap_{fixed_param_str_filename_model_chunk}.png")

                 # Call the new heatmap function
                 create_model_vs_chunk_overlap_heatmap(
                     data=filtered_df_for_model_chunk_test, # Pass the filtered data
                     output_path=test_output_path_model_chunk,
                     fixed_params=fixed_params_model_chunk, # Pass the fixed params for the title
                     values_col='f1_score',           # Explicitly state the value to plot
                     index_col_chunk='chunk_size',    # Rows = chunk size (part 1)
                     index_col_overlap='overlap_size',# Rows = overlap size (part 2)
                     columns_col='question_model'     # Columns = model
                 )
            else:
                 print(f"No data found for create_model_vs_chunk_overlap_heatmap test matching fixed parameters: {fixed_params_model_chunk}.")
                 print("Ensure your 'results' directory contains 'english' language files for the specified algorithm ('embedding') with varying models, chunk sizes, and overlap sizes.")

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