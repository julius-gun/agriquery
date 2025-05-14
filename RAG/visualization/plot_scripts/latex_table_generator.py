# RAG/visualization/plot_scripts/latex_table_generator.py
import os
import sys
import argparse
import pandas as pd

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

def generate_english_hybrid_latex_table(
    df_data: pd.DataFrame,
    output_dir: str,
    model_sort_order: list,
    metrics_to_include: list,
    filename: str = "english_hybrid_performance.tex",
    caption: str = "Performance of English Hybrid Models",
    label: str = "tab:english_hybrid_performance"
):
    """
    Generates a LaTeX table for English hybrid results, with bolded max values
    and formatting suitable for two-column layouts.

    Args:
        df_data (pd.DataFrame): The input DataFrame from extract_detailed_visualization_data.
        output_dir (str): Directory to save the .tex file.
        model_sort_order (list): List of model names to define row order.
        metrics_to_include (list): List of metric types (e.g., 'accuracy', 'f1_score') for columns.
        filename (str): Output filename for the .tex file.
        caption (str): Caption for the LaTeX table.
        label (str): Label for referencing the LaTeX table.
    """
    print(f"\n--- Generating LaTeX Table: {caption} ---")

    if df_data is None or df_data.empty:
        print("Error: Input DataFrame is empty. Cannot generate table.")
        return

    # 1. Filter data: English language, hybrid algorithm, and specified metrics
    df_filtered = df_data[
        (df_data['language'].str.lower() == 'english') &
        (df_data['retrieval_algorithm'].str.lower() == 'hybrid') &
        (df_data['metric_type'].isin(metrics_to_include))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_filtered.empty:
        print("Error: No data found for English hybrid models with the specified metrics. Cannot generate table.")
        print(f"Filters applied: language='english', retrieval_algorithm='hybrid', metrics={metrics_to_include}")
        # You might want to print some info about the original df_data for debugging
        # print(f"Original data languages: {df_data['language'].unique()}")
        # print(f"Original data retrieval_algorithms: {df_data['retrieval_algorithm'].unique()}")
        # print(f"Original data metric_types: {df_data['metric_type'].unique()}")
        return

    # 2. Pivot the table: models as rows, metrics as columns
    # We assume one result per model/metric_type combination for hybrid English.
    # If duplicates exist, pivot_table with an aggfunc (e.g., 'mean') might be needed.
    try:
        pivot_df = df_filtered.pivot_table(
            index='question_model',
            columns='metric_type',
            values='metric_value'
        )
    except Exception as e:
        print(f"Error pivoting data: {e}")
        print("Relevant data for pivot:")
        print(df_filtered[['question_model', 'metric_type', 'metric_value']].head())
        # Check for duplicates that might cause pivot issues without aggfunc
        duplicates = df_filtered[df_filtered.duplicated(subset=['question_model', 'metric_type'], keep=False)]
        if not duplicates.empty:
            print("Potential duplicate entries for (question_model, metric_type):")
            print(duplicates[['question_model', 'metric_type', 'metric_value', 'filename']])
        return

    # 3. Reorder rows (models) to include all from sort order, filling missing with NaN
    #    And select/reorder columns (metrics)
    
    # Ensure pivot_df is reindexed with the full model_sort_order to include all models,
    # filling those not in data with NaNs. This makes sure all desired rows appear.
    pivot_df = pivot_df.reindex(model_sort_order) # Use the full sort order
    
    # Filter out models from pivot_df that were not in the original data AND not in model_sort_order
    # (though reindex with model_sort_order should handle this correctly by keeping only those in model_sort_order)
    # Keep only models that are in model_sort_order. If a model was in data but not sort_order, it's dropped.
    # If a model was in sort_order but not data, it's kept (with NaNs).
    existing_models_in_sort_order = [model for model in model_sort_order if model in pivot_df.index]
    pivot_df = pivot_df.loc[existing_models_in_sort_order]


    final_metrics_columns = [metric for metric in metrics_to_include if metric in pivot_df.columns]
    missing_metrics_cols = [metric for metric in metrics_to_include if metric not in pivot_df.columns]
    if missing_metrics_cols:
        print(f"Warning: Metrics not found in pivoted data: {missing_metrics_cols}")
    
    # Ensure pivot_df only has the desired metric columns in the specified order
    pivot_df = pivot_df[final_metrics_columns]


    # Create a new DataFrame for LaTeX formatting. At this point, its index and columns match pivot_df.
    df_for_latex = pivot_df.copy()

    # 4. Apply Bold Formatting to Max Values and String Format Numeric Cells
    # This manipulation happens on df_for_latex, using pivot_df for numeric checks and max_val.
    # df_for_latex.index is IDENTICAL to pivot_df.index at this stage.
    for metric_col_name in final_metrics_columns: # These are original names like 'accuracy'
        if metric_col_name in df_for_latex.columns and pd.api.types.is_numeric_dtype(pivot_df[metric_col_name]):
            # Get max value from the original numeric pivot_df for accuracy
            max_val = pivot_df[metric_col_name].max()
            
            # Create the formatted series using pivot_df's data
            formatted_series = pivot_df[metric_col_name].apply(
                lambda x: f"\\textbf{{{x:.3f}}}" if pd.notna(x) and x == max_val 
                          else (f"{x:.3f}" if pd.notna(x) else "-")
            )
            # Assign this formatted series to df_for_latex. Indices match, so direct assignment works.
            df_for_latex[metric_col_name] = formatted_series
        elif metric_col_name in df_for_latex.columns: # Non-numeric column in original pivot_df
             # Convert these to string in df_for_latex, ensuring they are not NaNs from failed previous step
             df_for_latex[metric_col_name] = df_for_latex[metric_col_name].astype(str).fillna("-")


    # 5. Sanitize Model Names in Index of df_for_latex for LaTeX display
    # This is done AFTER all data cell manipulations are complete.
    df_for_latex.index = df_for_latex.index.str.replace('_', r'\_', regex=False)

    # 6. Rename Columns of df_for_latex for Display
    short_column_rename_map = {
        'f1_score': 'F1',
        'accuracy': 'Acc.',
        'precision': 'Prec.',
        'recall': 'Rec.',
        'specificity': 'Spec.'
        # Add other mappings if needed, ensure all in final_metrics_columns are covered
    }
    # Create a new list of renamed column headers for to_latex,
    # as df_for_latex.columns are still original, and we apply renaming via to_latex's header argument
    # Or, rename df_for_latex.columns directly:
    df_for_latex.rename(columns=lambda c: short_column_rename_map.get(c, c.replace('_', ' ').title()), inplace=True)


    # 7. Generate LaTeX string
    num_metric_cols_final = len(df_for_latex.columns) # Number of columns after processing
    col_format = "l" + "c" * num_metric_cols_final

    try:
        latex_string = df_for_latex.to_latex(
            column_format=col_format,
            # float_format="%.3f", # Not needed, cells are already strings
            caption=caption,
            label=label,
            header=True,
            index=True,
            escape=False, # CRITICAL: We have manually inserted LaTeX commands
            position='!htbp',
            na_rep="-" # Explicitly set how NaNs (if any somehow remain) are represented
        )
        
        # 8. Modify LaTeX String for Width: Inject \footnotesize
        # Pandas to_latex adds \centering by default if caption and label are used.
        # We want to insert \footnotesize after \centering.
        lines = latex_string.splitlines()
        inserted_footnotesize = False
        # Find centering and insert after, or before tabular if no centering
        centering_idx = -1
        tabular_idx = -1

        for i, line in enumerate(lines):
            if r'\centering' in line:
                centering_idx = i
                break # Prioritize inserting after centering
        
        if centering_idx != -1:
            lines.insert(centering_idx + 1, r'  \footnotesize')
            inserted_footnotesize = True
        else: # Fallback: if no \centering, insert before \begin{tabular}
            for i, line in enumerate(lines):
                if r'\begin{tabular}' in line:
                    tabular_idx = i
                    break
            if tabular_idx != -1:
                lines.insert(tabular_idx, r'  \footnotesize') # Insert before \begin{tabular}
                inserted_footnotesize = True
        
        if not inserted_footnotesize:
            print("Warning: Could not automatically inject \\footnotesize. Table might be too wide.")

        latex_string = '\n'.join(lines)

        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(latex_string)
        print(f"Successfully generated LaTeX table: {output_path}")
        print("Remember to include '\\usepackage{booktabs}' in your main LaTeX document.")
        if inserted_footnotesize:
            print("Added '\\footnotesize' for potentially better fit in two-column layouts.")

    except Exception as e:
        print(f"Error generating LaTeX string or writing file: {e}")
        print("DataFrame that was to be converted to LaTeX (df_for_latex):")
        print(df_for_latex.head())


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for English hybrid RAG test results."
    )
    # Default paths relative to PROJECT_ROOT
    default_results_path = os.path.join(PROJECT_ROOT, "results")
    # Output directory for LaTeX tables, can be within visualization/plots or a new dir
    default_output_path = os.path.join(PROJECT_ROOT, "visualization", "latex_tables")

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
    # Config path might be needed if extract_detailed_visualization_data relies on it for languages,
    # but it seems to have a fallback. Let's keep it for consistency if needed later.
    # default_config_path = os.path.join(PROJECT_ROOT, "config.json")
    # parser.add_argument(
    #     "--config-path",
    #     type=str,
    #     default=default_config_path,
    #     help="Path to the configuration JSON file (if needed by extractor).",
    # )

    args = parser.parse_args()

    print("--- Starting LaTeX Table Generation ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory for .tex files: {args.output_dir}")

    # 1. Extract Data
    print(f"\nExtracting detailed data from: {args.results_dir}")
    # Assuming extract_detailed_visualization_data does not strictly need config_path
    # for basic operation if languages are hardcoded/discovered.
    df_data = extract_detailed_visualization_data(args.results_dir)

    if df_data is None or df_data.empty:
        print("Exiting: No data extracted or DataFrame is empty.")
        return

    # --- Define metrics for the table ---
    metrics_for_table = ['accuracy', 'f1_score', 'precision', 'recall', 'specificity']

    # --- Generate the specific table ---
    generate_english_hybrid_latex_table(
        df_data=df_data,
        output_dir=args.output_dir,
        model_sort_order=REPORT_MODEL_SORT_ORDER,
        metrics_to_include=metrics_for_table
        # filename, caption, label use defaults
    )

    print("\n--- LaTeX Table Generation Finished ---")

if __name__ == "__main__":
    main()