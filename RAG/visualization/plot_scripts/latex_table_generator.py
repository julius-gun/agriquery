# RAG/visualization/plot_scripts/latex_table_generator.py
import os
import sys
import argparse
import pandas as pd
import json # Added for loading config

# --- Adjust Python Path (using the helper from plot_utils) ---
try:
    # This import will also execute the path setup code in plot_utils
    # because it runs at the module level there.
    from plot_utils import add_project_paths
    PROJECT_ROOT = add_project_paths() # Ensure paths are set and get the root
except ImportError as e:
    # Attempt to add paths manually if plot_utils is not found (e.g. running script directly)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.dirname(current_script_dir)
    project_root_fallback = os.path.dirname(visualization_dir)
    if project_root_fallback not in sys.path:
        sys.path.insert(0, project_root_fallback)
    PROJECT_ROOT = project_root_fallback # Use fallback root
    print(
        f"Warning: Could not import 'add_project_paths' from 'plot_utils.py'. "
        f"Attempted to set up paths manually. PROJECT_ROOT: {PROJECT_ROOT}"
    )
    print(f"Original error: {e}")
    # Proceed with caution, imports below might fail if paths are not correctly set

# --- Imports ---
try:
    from visualization.visualization_data_extractor import extract_detailed_visualization_data
    # Assuming main_visualization.py is in the same directory or accessible via path
    from main_visualization import REPORT_MODEL_SORT_ORDER
except ImportError as e:
    print("Error importing required modules after attempting path setup.")
    print(
        "This might indicate an issue with the project structure or missing files "
        "within 'utils' or 'visualization'."
    )
    print(f"Project root used: {PROJECT_ROOT}")
    print(f"Current sys.path: {sys.path}")
    print(f"Original Error: {e}")
    sys.exit(1)

def _generate_single_latex_table_string(
    df_data: pd.DataFrame,
    language: str,
    retrieval_algorithm: str,
    model_sort_order: list,
    metrics_to_include: list
) -> str | None:
    """
    Generates a LaTeX table string for a specific language and retrieval algorithm.

    Args:
        df_data (pd.DataFrame): The input DataFrame.
        language (str): The language to filter for (e.g., 'english').
        retrieval_algorithm (str): The retrieval algorithm to filter for (e.g., 'hybrid').
        model_sort_order (list): List of model names to define row order.
        metrics_to_include (list): List of metric types for columns.

    Returns:
        str | None: The LaTeX table string, or None if no data is found or an error occurs.
    """
    caption = f"Performance of {language.title()} {retrieval_algorithm.title()} Models"
    label = f"tab:{language.lower()}_{retrieval_algorithm.lower()}_performance"

    print(f"\n--- Generating LaTeX Table Snippet: {caption} ---")

    if df_data is None or df_data.empty:
        # This case should ideally be caught before calling this function per retrieval_algo/language.
        # However, good to have a safeguard.
        print(f"Warning: Input DataFrame is globally empty. Cannot generate table for {language} {retrieval_algorithm}.")
        return None

    # 1. Filter data
    df_filtered = df_data[
        (df_data['language'].str.lower() == language.lower()) &
        (df_data['retrieval_algorithm'].str.lower() == retrieval_algorithm.lower()) &
        (df_data['metric_type'].isin(metrics_to_include))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_filtered.empty:
        print(f"Info: No data found for {language.title()} {retrieval_algorithm.title()} with the specified metrics. Skipping table generation for this combination.")
        return None

    # 2. Pivot the table
    try:
        pivot_df = df_filtered.pivot_table(
            index='question_model',
            columns='metric_type',
            values='metric_value'
        )
    except Exception as e:
        print(f"Error pivoting data for {language.title()} {retrieval_algorithm.title()}: {e}")
        print("Relevant data for pivot:")
        print(df_filtered[['question_model', 'metric_type', 'metric_value']].head())
        duplicates = df_filtered[df_filtered.duplicated(subset=['question_model', 'metric_type'], keep=False)]
        if not duplicates.empty:
            print("Potential duplicate entries for (question_model, metric_type):")
            print(duplicates[['question_model', 'metric_type', 'metric_value', 'filename']])
        return None

    # 3. Reorder rows (models) and select/reorder columns (metrics)
    pivot_df = pivot_df.reindex(model_sort_order) 
    
    # Filter out models that are not in the reindexed pivot_df's index
    # (i.e., models in model_sort_order but not in original data are kept as NaN rows by reindex)
    # This step ensures we only operate on models that are supposed to be in the table.
    existing_models_in_sort_order = [model for model in model_sort_order if model in pivot_df.index]
    
    if not existing_models_in_sort_order:
        print(f"Warning: No models from model_sort_order found in the pivoted data for {language.title()} {retrieval_algorithm.title()}. Skipping table.")
        return None
    pivot_df = pivot_df.loc[existing_models_in_sort_order] # Ensures correct order and selection

    # Handle cases where all values for a model might be NaN after reindexing (e.g. model in sort order but no data)
    pivot_df.dropna(axis=0, how='all', inplace=True)
    if pivot_df.empty:
        print(f"Info: All models for {language.title()} {retrieval_algorithm.title()} had no data after filtering and reordering. Skipping table.")
        return None


    final_metrics_columns = [metric for metric in metrics_to_include if metric in pivot_df.columns]
    missing_metrics_cols = [metric for metric in metrics_to_include if metric not in pivot_df.columns]
    if missing_metrics_cols:
        print(f"Warning: For {language.title()} {retrieval_algorithm.title()}, metrics not found in pivoted data: {missing_metrics_cols}")
    
    if not final_metrics_columns:
        print(f"Warning: No metric columns to display for {language.title()} {retrieval_algorithm.title()} after filtering. Skipping table.")
        return None
        
    pivot_df = pivot_df[final_metrics_columns]


    df_for_latex = pivot_df.copy()

    # 4. Apply Bold Formatting to Max Values and String Format Numeric Cells
    for metric_col_name in final_metrics_columns: 
        if metric_col_name in df_for_latex.columns and pd.api.types.is_numeric_dtype(pivot_df[metric_col_name]):
            # Get max value from the original numeric pivot_df for accuracy
            max_val = pivot_df[metric_col_name].max() # Max of the potentially NaN-containing column
            
            # Create the formatted series using pivot_df's data
            formatted_series = pivot_df[metric_col_name].apply(
                lambda x: f"\\textbf{{{x:.3f}}}" if pd.notna(x) and x == max_val 
                          else (f"{x:.3f}" if pd.notna(x) else "-")
            )
            df_for_latex[metric_col_name] = formatted_series
        elif metric_col_name in df_for_latex.columns: # Non-numeric column in original pivot_df (should not happen with metrics)
             df_for_latex[metric_col_name] = df_for_latex[metric_col_name].astype(str).fillna("-")


    # 5. Sanitize Model Names in Index of df_for_latex for LaTeX display
    df_for_latex.index = df_for_latex.index.str.replace('_', r'\_', regex=False)

    # 6. Rename Columns of df_for_latex for Display
    short_column_rename_map = {
        'f1_score': 'F1',
        'accuracy': 'Acc.',
        'precision': 'Prec.',
        'recall': 'Rec.',
        'specificity': 'Spec.'
        # Add other mappings if needed
    }
    df_for_latex.rename(columns=lambda c: short_column_rename_map.get(c, c.replace('_', ' ').title()), inplace=True)


    # 7. Generate LaTeX string
    num_metric_cols_final = len(df_for_latex.columns) 
    col_format = "l" + "c" * num_metric_cols_final # e.g. "lcc" for one model name col, two metric cols

    try:
        latex_string = df_for_latex.to_latex(
            column_format=col_format,
            caption=caption,
            label=label,
            header=True, # Keep column headers
            index=True,  # Keep model names as index
            escape=False, # CRITICAL: We have manually inserted LaTeX commands (e.g. \textbf, \_)
            position='!htbp', # Suggested position for float
            na_rep="-" # Representation for any remaining NaNs (should be handled by formatting step)
        )
        
        # 8. Modify LaTeX String for Width: Inject \footnotesize
        lines = latex_string.splitlines()
        inserted_footnotesize = False
        centering_idx = -1
        tabular_idx = -1

        for i, line in enumerate(lines):
            if r'\centering' in line:
                centering_idx = i
                break 
        
        if centering_idx != -1:
            lines.insert(centering_idx + 1, r'  \footnotesize')
            inserted_footnotesize = True
        else: 
            for i, line in enumerate(lines):
                if r'\begin{tabular}' in line:
                    tabular_idx = i
                    break
            if tabular_idx != -1:
                lines.insert(tabular_idx, r'  \footnotesize') 
                inserted_footnotesize = True
        
        if not inserted_footnotesize: # Should always find one of them if to_latex works as expected
            print(f"Warning: Could not automatically inject \\footnotesize for {language.title()} {retrieval_algorithm.title()}. Table might be too wide.")

        latex_string = '\n'.join(lines)
        
        print(f"Successfully generated LaTeX table snippet for {language.title()} {retrieval_algorithm.title()}.")
        return latex_string

    except Exception as e:
        print(f"Error generating LaTeX string for {language.title()} {retrieval_algorithm.title()}: {e}")
        print("DataFrame that was to be converted to LaTeX (df_for_latex):")
        print(df_for_latex.head())
        return None


def generate_latex_reports_by_method(
    df_data: pd.DataFrame,
    output_dir: str,
    model_sort_order: list,
    metrics_to_include: list,
    config_data: dict
):
    """
    Generates .tex files, one for each specified retrieval method ("hybrid", "embedding").
    Each file contains tables for all configured languages for that method.
    """
    if df_data is None or df_data.empty:
        print("Error: Input DataFrame is empty. Cannot generate any reports.")
        return

    # Get languages from config
    languages_from_config = [lang_conf["language"] for lang_conf in config_data.get("language_configs", [])]
    if not languages_from_config:
        # Fallback to languages present in data if config doesn't specify any
        print("Warning: No languages found in config. Using distinct languages from data.")
        languages_from_config = sorted(list(df_data['language'].str.lower().unique()))
        if not languages_from_config:
            print("Error: No languages found in config or data. Cannot generate reports.")
            return
    
    # Retrieval methods to process, as specifically requested
    retrieval_methods_to_process = ["hybrid", "embedding", "keyword"]
    
    # Filter this list to only those methods actually present in the data
    available_retrieval_methods_in_data = df_data['retrieval_algorithm'].str.lower().unique()
    actual_methods_to_process = [
        method for method in retrieval_methods_to_process
        if method in available_retrieval_methods_in_data
    ]

    if not actual_methods_to_process:
        print(f"Warning: None of the target retrieval methods ({retrieval_methods_to_process}) found in data. "
              f"Available methods in data: {list(available_retrieval_methods_in_data)}. No .tex files will be generated.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for retrieval_method in actual_methods_to_process:
        print(f"\n--- Processing Retrieval Method: {retrieval_method.title()} ---")
        all_tables_for_this_method_latex = []

        for lang in languages_from_config:
            table_latex_string = _generate_single_latex_table_string(
                df_data=df_data,
                language=lang,
                retrieval_algorithm=retrieval_method,
                model_sort_order=model_sort_order,
                metrics_to_include=metrics_to_include
            )
            if table_latex_string:
                all_tables_for_this_method_latex.append(table_latex_string)
        
        if all_tables_for_this_method_latex:
            # Combine all table strings for this retrieval method
            # Using a LaTeX comment and double newline as a separator.
            # \clearpage or \newpage could be used if each table should start on a new page.
            separator = f"\n\n% Table for next language using {retrieval_method} retrieval\n\n"
            combined_latex_content = separator.join(all_tables_for_this_method_latex)
            
            output_filename = f"{retrieval_method}_all_languages_performance.tex"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                with open(output_path, "w", encoding='utf-8') as f:
                    f.write(combined_latex_content)
                print(f"Successfully generated consolidated LaTeX file: {output_path}")
            except IOError as e:
                print(f"Error writing file {output_path}: {e}")
        else:
            print(f"No tables generated for retrieval method: {retrieval_method.title()}. No .tex file created for it.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for RAG test results, grouped by retrieval method."
    )
    default_results_path = os.path.join(PROJECT_ROOT, "results")
    default_output_path = os.path.join(PROJECT_ROOT, "visualization", "latex_tables")
    # Default config path should point to the root of the project for config.json
    default_config_path = os.path.join(PROJECT_ROOT, "config.json") 

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
        help="Directory where the generated .tex files will be saved.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=default_config_path,
        help="Path to the configuration JSON file.",
    )

    args = parser.parse_args()

    print("--- Starting LaTeX Table Generation ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory for .tex files: {args.output_dir}")
    print(f"Config file: {args.config_path}")

    # 0. Load Config
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_path}. Cannot determine languages or other settings.")
        sys.exit(1) # Exit if config is essential and not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config_path}. Check its format.")
        sys.exit(1) # Exit if config is malformed


    # 1. Extract Data
    # Assuming extract_detailed_visualization_data might use config for mapping or other details.
    # If it solely relies on files in results_dir, config_path might not be strictly needed for it.
    # However, it's good practice to pass it if it *could* use it.
    # For this refactoring, extract_detailed_visualization_data's signature is not changed.
    print(f"\nExtracting detailed data from: {args.results_dir}")
    df_data = extract_detailed_visualization_data(args.results_dir) 

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        return

    # --- Define metrics for the table ---
    metrics_for_table = ['accuracy', 'f1_score', 'precision', 'recall', 'specificity']

    # --- Generate the tables ---
    generate_latex_reports_by_method(
        df_data=df_data,
        output_dir=args.output_dir,
        model_sort_order=REPORT_MODEL_SORT_ORDER, # From main_visualization
        metrics_to_include=metrics_for_table,
        config_data=config_data # Pass loaded config
    )
    
    print("\nReminder: Ensure your main LaTeX document includes '\\usepackage{booktabs}'.")
    print("If using \\footnotesize, \\usepackage{graphicx} might be needed for some scaling options, though usually not for \\footnotesize itself.")
    print("\n--- LaTeX Table Generation Finished ---")

if __name__ == "__main__":
    main()