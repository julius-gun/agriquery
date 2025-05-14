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
    Generates a LaTeX table for English hybrid results.

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

    # 3. Reorder rows (models) based on model_sort_order
    # And select only models present in the data to avoid KeyError
    ordered_models = [model for model in model_sort_order if model in pivot_df.index]
    missing_models = [model for model in model_sort_order if model not in pivot_df.index]
    if missing_models:
        print(f"Warning: The following models from sort order were not found in the data: {missing_models}")
    
    pivot_df = pivot_df.reindex(ordered_models)

    # 4. Select and reorder columns (metrics) based on metrics_to_include
    # Ensure only metrics present in the pivot_df are selected
    final_metrics_columns = [metric for metric in metrics_to_include if metric in pivot_df.columns]
    missing_metrics_cols = [metric for metric in metrics_to_include if metric not in pivot_df.columns]
    if missing_metrics_cols:
        print(f"Warning: The following metrics were not found in the pivoted data: {missing_metrics_cols}")

    pivot_df = pivot_df[final_metrics_columns]

    # Clean up model names for display (optional, e.g., replace underscores)
    # pivot_df.index = pivot_df.index.str.replace('_', ' ').str.title()
    
    # Clean up column names for display (e.g., F1 Score instead of f1_score)
    column_rename_map = {
        'f1_score': 'F1 Score',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'specificity': 'Specificity'
    }
    pivot_df.rename(columns=lambda c: column_rename_map.get(c, c.replace('_', ' ').title()), inplace=True)


    # 5. Generate LaTeX string
    # Using a fixed number of columns (len(metrics_to_include) + 1 for model name)
    # 'l' for model name (left-aligned), 'c' for metrics (center-aligned)
    num_metric_cols = len(final_metrics_columns)
    col_format = "l" + "c" * num_metric_cols

    try:
        latex_string = pivot_df.to_latex(
            column_format=col_format,
            float_format="%.3f",  # Format numbers to 3 decimal places
            caption=caption,
            label=label,
            header=True, # Keep header (metric names)
            index=True, # Keep index (model names)
            escape=False, # Assuming model names and headers don't need LaTeX escaping
                          # Set to True if they contain special LaTeX characters like _
            position='!htbp' # Suggested LaTeX float position
        )
        
        # Add \usepackage{booktabs} if using it (to_latex does by default)
        # and adjust table environment for better spacing if needed
        # For example, ensure we have a complete document structure or just the table part.
        # Pandas to_latex by default includes \begin{table}...\end{table}
        # and often \usepackage{booktabs} comments.

        # A common practice is to wrap with a simple document for testing.
        # For integration, you might just want the table content itself.
        # The output of to_latex() is usually self-contained for a table.

        # Prepend \usepackage{booktabs} if not already included by user's main .tex file
        # to_latex() uses \toprule, \midrule, \bottomrule which require booktabs.
        # A common approach is to assume the user's document has it.
        # For robustness, we could add a comment suggesting it.
        
        # latex_output = "\\documentclass{article}\n"
        # latex_output += "\\usepackage{booktabs} % Required for to_latex table rules\n"
        # latex_output += "\\usepackage{caption} % For better caption control\n"
        # latex_output += "\\begin{document}\n"
        # latex_output += latex_string
        # latex_output += "\\end{document}\n"
        # For now, just save the direct output of to_latex()

        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(latex_string)
        print(f"Successfully generated LaTeX table: {output_path}")
        print("Remember to include '\\usepackage{booktabs}' in your main LaTeX document for optimal table rendering.")

    except Exception as e:
        print(f"Error generating LaTeX string or writing file: {e}")
        print("Pivoted DataFrame that was to be converted:")
        print(pivot_df.head())


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