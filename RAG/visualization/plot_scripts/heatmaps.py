# visualization/plot_scripts/heatmaps.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Optional, Tuple, List # Added List

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

    # --- Apply Optional Filtering ---
    filtered_data = data.copy()
    filter_description = "all data"
    if filter_params:
        filter_items = []
        print(f"Applying filter: {filter_params}") # Debug print
        for key, value in filter_params.items():
            if key in filtered_data.columns:
                print(f"  Filtering by {key} == {value}...") # Debug print
                original_count = len(filtered_data)
                filtered_data = filtered_data[filtered_data[key] == value]
                print(f"    Count before: {original_count}, Count after: {len(filtered_data)}") # Debug print
                filter_items.append(f"{key}={value}")
            else:
                print(f"Warning: Filter key '{key}' not found in DataFrame columns. Ignoring.")
        if not filtered_data.empty:
             filter_description = ", ".join(filter_items)
             print(f"Data filtered successfully. Filter description: {filter_description}") # Debug print
        else:
            print(f"Warning: Filtering with {filter_params} resulted in an empty DataFrame. Cannot create heatmap with data.")
            # We might still proceed if all_indices is set, to create a blank heatmap structure
            # Let's create an empty pivot table structure to allow reindexing
            pivot_df = pd.DataFrame(columns=data[columns_col].unique(), index=pd.Index([], name=index_col))

    # --- Pivot Data ---
    if not filtered_data.empty: # Only pivot if there's data after filtering
        try:
            # Use pivot_table with mean aggregation in case multiple runs match the filter
            # (though ideally the filter should isolate a unique set)
            print("Creating pivot table...") # Debug print
            pivot_df = pd.pivot_table(
                filtered_data,
                values=values_col,
                index=index_col,
                columns=columns_col,
                aggfunc='mean' # Use mean if duplicates exist for a given index/column pair after filtering
            )
            print("Pivot table created.") # Debug print
            # print(pivot_df.head()) # Debug print - potentially large output
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            return
    # If filtered_data was empty, pivot_df was initialized as an empty structure above

    # --- Ensure all expected indices are present ---
    if all_indices:
        print(f"Ensuring heatmap index includes: {all_indices}")
        # Reindex the pivot table. Missing indices will be added with NaN values.
        pivot_df = pivot_df.reindex(all_indices)
        print("Pivot table after reindexing.") # Debug print
        # print(pivot_df.head()) # Debug print - potentially large output
        # Optional: Sort index if desired (e.g., alphabetically)
        # pivot_df = pivot_df.sort_index()

    # Check if the pivot table is effectively empty (all NaNs) even after reindexing
    if pivot_df.isnull().all().all():
         print(f"Warning: Pivot table contains only NaN values after filtering and reindexing. Heatmap will be blank or show NaN markers.")
         # Proceed to plot the blank/NaN heatmap, as it indicates no data found for the filter + indices.

    # --- Plotting ---
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set plot style
    sns.set_theme(style="white")

    # Create the figure
    plt.figure(figsize=figsize)

    # Generate the heatmap
    print("Generating heatmap...") # Debug print
    ax = sns.heatmap(
        pivot_df,
        annot=True,          # Show values in cells
        fmt=annot_fmt,       # Format for the annotations
        cmap=cmap,           # Color map
        linewidths=.5,       # Add lines between cells
        linecolor='lightgray', # Color of the lines
        cbar=True,           # Show the color bar
        vmin=0.0,            # Minimum value for color scale (F1 score range)
        vmax=1.0,            # Maximum value for color scale (F1 score range)
        # na_color='whitesmoke' # REMOVED: Invalid argument
    )

    # Improve layout and labels
    plt.xticks(rotation=45, ha='right') # Rotate column labels for readability
    plt.yticks(rotation=0) # Keep row labels horizontal

    # Set title and labels
    default_title = f'{values_col.replace("_", " ").title()} Comparison ({index_col.title()} vs {columns_col.title()})'
    if filter_params:
        default_title += f'\n(Filter: {filter_description})'
    plot_title = title or default_title
    x_axis_label = xlabel or columns_col.replace("_", " ").title()
    y_axis_label = ylabel or index_col.replace("_", " ").title()

    plt.title(plot_title, fontsize=14, pad=20) # Add padding to title
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)

    plt.tight_layout() # Adjust layout

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight') # Use bbox_inches='tight'
        print(f"Heatmap saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving heatmap to {output_path}: {e}")

    plt.close() # Close the plot figure

# Example usage (if run directly, though typically called from main_visualization.py)
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
            test_output_path = os.path.join(test_output_dir, "test_f1_heatmap_lang_vs_model_reindexed.png")

            # Define a filter (e.g., specific chunk size, overlap, top_k, algo)
            # Adjust these values based on your actual result filenames
            test_filter = {
                'retrieval_algorithm': 'embedding',
                'chunk_size': 2000,
                'overlap_size': 100,
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
                # Optional: customize title, labels, etc.
                # title="F1 Score: Language vs. Question Model (Embedding, CS=2000, OS=100, K=3)"
            )
        else:
            print("Could not load example data. Ensure 'results' directory exists and contains valid files.")
            print("Or run 'main_visualization.py' instead.")

    except ImportError as e:
        print(f"Could not import necessary modules (extract_visualization_data, ConfigLoader). Run 'main_visualization.py' for full functionality. Error: {e}")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")

    print("--- Heatmap Creation Test Finished ---")
