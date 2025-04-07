# visualization/plot_scripts/heatmaps.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Optional, Tuple, List # Added List

# --- DEBUG FLAG ---
# Set to True to enable detailed printing, False to disable
DEBUG_HEATMAP = True

def create_f1_heatmap(
    data: pd.DataFrame,
    output_path: str,
    index_col: str = 'language',
    columns_col: str = 'question_model',
    values_col: str = 'f1_score',
    all_indices: Optional[List[str]] = None, # New parameter for all expected index values (e.g., all languages)
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    annot_fmt: str = ".3f",
    figsize: Tuple[int, int] = (14, 8), # Adjusted default size
    filter_params: Optional[dict] = None # Optional dict to filter data before pivoting
):
    """
    Creates a heatmap of F1 scores, typically with languages as rows and models as columns.
    Ensures all specified indices (e.g., languages) are included, even if data is missing.

    Args:
        data: Pandas DataFrame containing the extracted results data.
              Must include columns specified by index_col, columns_col, and values_col.
        output_path: The full path where the plot image will be saved.
        index_col: The column name to use for the heatmap rows (index). Default: 'language'.
        columns_col: The column name to use for the heatmap columns. Default: 'question_model'.
        values_col: The column name containing the scores for the heatmap cells. Default: 'f1_score'.
        all_indices: Optional list of all expected values for the index (e.g., ['english', 'french', 'german']).
                     If provided, the heatmap index will be forced to include all these values,
                     filling missing data with NaN. Default: None (uses only indices present in data).
        title: Optional title for the plot. If None, a default title is generated.
        xlabel: Optional label for the x-axis (columns). If None, uses columns_col.
        ylabel: Optional label for the y-axis (rows). If None, uses index_col.
        cmap: Colormap to use for the heatmap (e.g., "viridis", "coolwarm", "YlGnBu").
        annot_fmt: String format for annotations within the heatmap cells. Default: ".3f".
        figsize: Tuple specifying the figure size (width, height) in inches.
        filter_params: Optional dictionary of parameters to filter the DataFrame *before* pivoting.
                       Example: {'chunk_size': 2000, 'retrieval_algorithm': 'embedding'}
                       This is useful if the input DataFrame contains results from multiple
                       configurations (e.g., different chunk sizes).
    """
    if data is None or data.empty:
        print("Error: Cannot create heatmap. Input data is None or empty.")
        return
    if not all(col in data.columns for col in [index_col, columns_col, values_col]):
        print(f"Error: DataFrame must contain columns: '{index_col}', '{columns_col}', '{values_col}'.")
        print(f"Available columns: {data.columns.tolist()}")
        return

    if DEBUG_HEATMAP:
        print("\n--- Heatmap Debug: Initial Data ---")
        # Print relevant columns for easier checking
        cols_to_print = [index_col, columns_col, values_col] + list(filter_params.keys() if filter_params else [])
        cols_to_print = list(set(col for col in cols_to_print if col in data.columns)) # Ensure columns exist
        print(data[cols_to_print].to_string())
        print("-" * 30)

    # --- Apply Optional Filtering ---
    filtered_data = data.copy()
    filter_description = "all data"
    if filter_params:
        filter_items = []
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Applying filter: {filter_params} ---")
        for key, value in filter_params.items():
            if key in filtered_data.columns:
                original_count = len(filtered_data)
                # Ensure type consistency for comparison if needed (e.g., int vs str)
                try:
                    # Attempt conversion if filter value type differs from column dtype
                    if pd.api.types.is_numeric_dtype(filtered_data[key]) and not isinstance(value, (int, float)):
                         filter_value = type(filtered_data[key].iloc[0])(value) # Convert filter value to column type
                    elif pd.api.types.is_string_dtype(filtered_data[key]) and not isinstance(value, str):
                         filter_value = str(value)
                    else:
                         filter_value = value
                    filtered_data = filtered_data[filtered_data[key] == filter_value]
                except Exception as e:
                    print(f"Warning: Error during type conversion for filter key '{key}' (value: {value}). Skipping filter. Error: {e}")
                    continue # Skip this filter key if conversion fails

                if DEBUG_HEATMAP: print(f"  Filtering by {key} == {filter_value} -> Count before: {original_count}, Count after: {len(filtered_data)}")
                filter_items.append(f"{key}={value}")
            else:
                print(f"Warning: Filter key '{key}' not found in DataFrame columns. Ignoring.")
        if not filtered_data.empty:
             filter_description = ", ".join(filter_items)
             if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Data after filtering ---\n{filtered_data[cols_to_print].to_string()}\n" + "-"*30)
        else:
            print(f"Warning: Filtering with {filter_params} resulted in an empty DataFrame. Cannot create heatmap with data.")
            pivot_df = pd.DataFrame(columns=data[columns_col].unique(), index=pd.Index([], name=index_col))

    # --- Pivot Data ---
    if not filtered_data.empty:
        try:
            if DEBUG_HEATMAP: print("--- Heatmap Debug: Creating pivot table ---")
            pivot_df = pd.pivot_table(
                filtered_data,
                values=values_col,
                index=index_col,
                columns=columns_col,
                aggfunc='mean'
            )
            if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Pivot table BEFORE reindexing ---\n{pivot_df.to_string()}\n" + "-"*30)
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            return
    elif not filter_params: # If no filter was applied and data was initially empty
         print("Error: Input data is empty and no filter was applied.")
         return
    # If filtered_data was empty due to filtering, pivot_df was initialized above

    # --- Ensure all expected indices are present ---
    if all_indices:
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Ensuring heatmap index includes: {all_indices} ---")
        pivot_df = pivot_df.reindex(all_indices)
        if DEBUG_HEATMAP: print(f"--- Heatmap Debug: Pivot table AFTER reindexing ---\n{pivot_df.to_string()}\n" + "-"*30)

    if pivot_df.isnull().all().all():
         print(f"Warning: Pivot table contains only NaN values after filtering and reindexing. Heatmap will be blank or show NaN markers.")

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

    default_title = f'{values_col.replace("_", " ").title()} Comparison ({index_col.title()} vs {columns_col.title()})'
    if filter_params:
        default_title += f'\n(Filter: {filter_description})'
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

# Example usage part remains the same as before...
if __name__ == '__main__':
    # This part is for demonstration; assumes extract_visualization_data is available
    # and you have some results files.
    print("--- Testing Heatmap Creation (Example) ---")

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
            test_output_path = os.path.join(test_output_dir, "test_f1_heatmap_lang_vs_model_reindexed_debug.png")

            # Define a filter (e.g., specific chunk size, overlap, top_k, algo)
            # !!! ADJUST THESE VALUES TO MATCH ONE OF YOUR RUNS !!!
            # Example: Filter for the phi4 run
            test_filter = {
                'retrieval_algorithm': 'embedding',
                'chunk_size': 2000,
                'overlap_size': 50, # Matches phi4 run
                'num_retrieved_docs': 3
            }
            print(f"Applying filter: {test_filter}")
            print(f"Using language list for reindexing: {all_languages_list}")

            # Create the heatmap, passing the language list
            create_f1_heatmap(
                data=example_df,
                output_path=test_output_path,
                filter_params=test_filter,
                all_indices=all_languages_list, # Pass the list here
                title="DEBUG Heatmap: F1 Score (Filter: Emb, CS=2000, OS=50, K=3)"
            )

            # Example: Filter for the phi3 run
            test_filter_2 = {
                'retrieval_algorithm': 'embedding',
                'chunk_size': 2000,
                'overlap_size': 100, # Matches phi3 run
                'num_retrieved_docs': 3
            }
            test_output_path_2 = os.path.join(test_output_dir, "test_f1_heatmap_lang_vs_model_reindexed_debug_phi3.png")
            print(f"\nApplying filter: {test_filter_2}")
            create_f1_heatmap(
                data=example_df,
                output_path=test_output_path_2,
                filter_params=test_filter_2,
                all_indices=all_languages_list,
                title="DEBUG Heatmap: F1 Score (Filter: Emb, CS=2000, OS=100, K=3)"
            )

        else:
            print("Could not load example data. Ensure 'results' directory exists and contains valid files.")
            print("Or run 'main_visualization.py' instead.")

    except ImportError as e:
        print(f"Could not import necessary modules (extract_visualization_data, ConfigLoader). Run 'main_visualization.py' for full functionality. Error: {e}")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")

    print("--- Heatmap Creation Test Finished ---")