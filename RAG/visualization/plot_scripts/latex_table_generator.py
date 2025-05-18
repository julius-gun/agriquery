# RAG/visualization/plot_scripts/latex_table_generator.py
import os
import sys
import argparse
import pandas as pd
import json # Added for loading config
import re # Added for model name abbreviation

# --- Constants for "Full Manual" ---
FULL_MANUAL_ALIAS = "Full Manual"
FULL_MANUAL_NOISE_LEVEL = 59000
FULL_MANUAL_INTERNAL_ID = "full_manual_report_type" # Internal identifier


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

def _abbreviate_model_name(model_name: str) -> str:
    """
    Abbreviates model names for display in LaTeX tables.
    e.g., 'gemini-2.5-flash-preview-04-17' -> 'gemini-2.5-flash'
          'phi3_14B_q4_medium-128k' -> 'phi3_14B'
          'deepseek-r1_8B-128k' -> 'deepseek-r1_8B'
    """
    # Remove preview dates like -preview-MM-DD or -preview-MM
    name = re.sub(r'-preview-\d{2}-\d{2}$', '', model_name)
    name = re.sub(r'-preview-\d{2}$', '', name) # For cases like -preview-05

    # Specific handling for phi3 models to remove quantization and context window
    if name.startswith("phi3"):
        # Retain the core part like phi3_14B, phi3_8B
        match = re.match(r'(phi3(?:_mini|_medium|_small)?(?:_\d+B)?)', name)
        if match:
            name = match.group(1)
            # Further cleanup for phi3: if it was 'phi3_14B_q4_medium-128k' -> 'phi3_14B_q4_medium'
            # then simplify to 'phi3_14B'
            # If it was 'phi3_mini-128k' -> 'phi3_mini'
            name = name.replace("_q4_medium", "").replace("_q4_mini", "").replace("_medium","").replace("_mini","")


    # General rule: remove context window suffixes like -128k, -8k, -4k, -16k
    # This regex tries to be specific to avoid removing parts of model names like 'qwen2.5_7B'
    name = re.sub(r'-\d+k$', '', name)

    # General rule: remove common ollama tag parts like ':latest', ':8b', etc.
    # This is often part of the 'name' field in config but might appear in question_model if not handled upstream
    name = re.sub(r':\w+$', '', name)
    name = re.sub(r':\d+(\.\d+)?[bB]$', '', name) # e.g. :8b, :1.5b


    # Further specific simplifications based on provided examples
    if "gemini-2.5-flash" in name: # Catches gemini-2.5-flash-preview...
        name = "Gemini 2.5 Flash"
    if "qwen2.5_7B" in name: # Catches qwen2.5_7B-128k
        name = "Qwen 2.5 7B"
    if "qwen3_8B" in name: # Catches qwen3_8B-128k
        name = "Qwen3 8B"
    if "phi3_14B" in name and ("medium" in model_name or "128k" in model_name or "4k" in model_name): # Catches phi3_14B_q4_medium-128k or phi3_14B_medium-4k
        name = "Phi3 14B"
    if "llama3.1_8B" in name: # Catches llama3.1_8B-128k
        name = "Llama3.1 8B"
    if "deepseek-r1_8B" in name: # Catches deepseek-r1_8B-128k
        name = "Deepseek-R1 8B"
    if "llama3.2_3B" in name: # Catches llama3.2_3B-128k
        name = "Llama3.2 3B"
    if "deepseek-r1_1.5B" in name: # Catches deepseek-r1_1.5B-128k
        name = "Deepseek-R1 1.5B"
    if "llama3.2_1B" in name: # Catches llama3.2_1B-128k
        name = "Llama3.2 1B"

    # Replace underscores with backslash-underscore for LaTeX compatibility AFTER abbreviation
    # This step is moved from later in the _generate_single_latex_table_string function
    # as it should apply to the final abbreviated name.
    # However, since we are doing a .str.replace later on the index, this might be redundant here
    # or can be applied to the index directly. For now, let's assume the index replacement is sufficient.

    return name

def _generate_single_latex_table_string(
    df_data: pd.DataFrame,
    language: str,
    retrieval_algorithm_or_report_type: str, # Renamed for clarity
    model_sort_order: list,
    metrics_to_include: list
) -> str | None:
    """
    Generates a LaTeX table string for a specific language and retrieval algorithm/report type.

    Args:
        df_data (pd.DataFrame): The input DataFrame.
        language (str): The language to filter for (e.g., 'english').
        retrieval_algorithm_or_report_type (str): The retrieval algorithm (e.g., 'hybrid')
                                                  or a special report type identifier
                                                  (e.g., FULL_MANUAL_INTERNAL_ID).
        model_sort_order (list): List of model names to define row order.
                                 These names should be the *original, unabbreviated* names.
        metrics_to_include (list): List of metric types for columns.

    Returns:
        str | None: The LaTeX table string, or None if no data is found or an error occurs.
    """
    display_name_for_caption = ""
    label_suffix = ""

    if retrieval_algorithm_or_report_type == FULL_MANUAL_INTERNAL_ID:
        display_name_for_caption = FULL_MANUAL_ALIAS
        label_suffix = "full_manual"
        print(f"\n--- Generating LaTeX Table Snippet for: {language.title()} {FULL_MANUAL_ALIAS} ---")
        # Specific filtering for Full Manual
        df_filtered = df_data[
            (df_data['language'].str.lower() == language.lower()) &
            (df_data['retrieval_algorithm'].str.lower() == "zeroshot") &
            (df_data['noise_level'] == FULL_MANUAL_NOISE_LEVEL) &
            (df_data['metric_type'].isin(metrics_to_include))
        ].copy()
    else:
        # Standard retrieval algorithm processing
        display_name_for_caption = retrieval_algorithm_or_report_type.title()
        label_suffix = retrieval_algorithm_or_report_type.lower()
        print(f"\n--- Generating LaTeX Table Snippet: {language.title()} {display_name_for_caption} Models ---")
        df_filtered = df_data[
            (df_data['language'].str.lower() == language.lower()) &
            (df_data['retrieval_algorithm'].str.lower() == retrieval_algorithm_or_report_type.lower()) &
            (df_data['metric_type'].isin(metrics_to_include))
        ].copy()

    caption = f"Performance of {language.title()} {display_name_for_caption} Models"
    label = f"tab:{language.lower()}_{label_suffix}_performance"


    if df_data is None or df_data.empty:
        # This case should ideally be caught before calling this function per retrieval_algo/language.
        # However, good to have a safeguard.
        print(f"Warning: Input DataFrame is globally empty. Cannot generate table for {language} {display_name_for_caption}.")
        return None

    if df_filtered.empty:
        print(f"Info: No data found for {language.title()} {display_name_for_caption} with the specified metrics. Skipping table generation for this combination.")
        return None

    # Abbreviate model names in the 'question_model' column
    df_filtered['question_model'] = df_filtered['question_model'].apply(_abbreviate_model_name)

    # Abbreviate model names in the sort order list
    # Create a mapping from original to abbreviated for consistent sorting if needed,
    # or just abbreviate the sort order directly.
    # Let's create a new list of abbreviated sort order models.
    # We need to preserve the order and uniqueness.
    abbreviated_model_sort_order_map = {original_name: _abbreviate_model_name(original_name) for original_name in model_sort_order}
    
    # Get unique abbreviated names in the desired order
    unique_abbreviated_model_sort_order = []
    seen_abbreviated_names = set()
    for original_name in model_sort_order:
        abbreviated_name = abbreviated_model_sort_order_map[original_name]
        if abbreviated_name not in seen_abbreviated_names:
            unique_abbreviated_model_sort_order.append(abbreviated_name)
            seen_abbreviated_names.add(abbreviated_name)


    # 2. Pivot the table using the now abbreviated 'question_model' names
    try:
        pivot_df = df_filtered.pivot_table(
            index='question_model', # This now contains abbreviated names
            columns='metric_type',
            values='metric_value'
        )
    except Exception as e:
        print(f"Error pivoting data for {language.title()} {display_name_for_caption}: {e}")
        print("Relevant data for pivot:")
        print(df_filtered[['question_model', 'metric_type', 'metric_value']].head())
        duplicates = df_filtered[df_filtered.duplicated(subset=['question_model', 'metric_type'], keep=False)]
        if not duplicates.empty:
            print("Potential duplicate entries for (question_model, metric_type):")
            print(duplicates[['question_model', 'metric_type', 'metric_value', 'filename']])
        return None

    # 3. Reorder rows (models) and select/reorder columns (metrics)
    pivot_df = pivot_df.reindex(unique_abbreviated_model_sort_order) 
    
    # Filter out models that are not in the reindexed pivot_df's index
    # (i.e., models in unique_abbreviated_model_sort_order but not in original data are kept as NaN rows by reindex)
    # This step ensures we only operate on models that are supposed to be in the table.
    existing_models_in_sort_order = [model for model in unique_abbreviated_model_sort_order if model in pivot_df.index]
    
    if not existing_models_in_sort_order:
        print(f"Warning: No models from model_sort_order found in the pivoted data for {language.title()} {display_name_for_caption}. Skipping table.")
        return None
    pivot_df = pivot_df.loc[existing_models_in_sort_order] # Ensures correct order and selection

    # Handle cases where all values for a model might be NaN after reindexing (e.g. model in sort order but no data)
    pivot_df.dropna(axis=0, how='all', inplace=True)
    if pivot_df.empty:
        print(f"Info: All models for {language.title()} {display_name_for_caption} had no data after filtering and reordering. Skipping table.")
        return None


    final_metrics_columns = [metric for metric in metrics_to_include if metric in pivot_df.columns]
    missing_metrics_cols = [metric for metric in metrics_to_include if metric not in pivot_df.columns]
    if missing_metrics_cols:
        print(f"Warning: For {language.title()} {display_name_for_caption}, metrics not found in pivoted data: {missing_metrics_cols}")
    
    if not final_metrics_columns:
        print(f"Warning: No metric columns to display for {language.title()} {display_name_for_caption} after filtering. Skipping table.")
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
    
    # Crucial step: Set index name to None to prevent to_latex from printing it as a header for the index column
    df_for_latex.index.name = None


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
    col_format = "l" + "c" * num_metric_cols_final 

    # Store renamed column headers for manual insertion.
    # These are already abbreviated, e.g., 'Acc.', 'F1', due to step 6.
    renamed_column_headers = list(df_for_latex.columns)

    try:
        latex_string = df_for_latex.to_latex(
            column_format=col_format,
            caption=caption,
            label=label,
            header=False, # Set to False: We will manually insert the desired header.
            index=True,
            escape=False, 
            position='!htbp', 
            na_rep="-" 
        )
        
        # 8. Modify LaTeX String
        lines = latex_string.splitlines()
        
        # 8a. Inject \footnotesize (existing logic, applied first)
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
        
        if not inserted_footnotesize: 
            print(f"Warning: Could not automatically inject \\footnotesize for {caption}. Table might be too wide.")

        # 8b. Inject custom two-line header
        toprule_idx = -1
        for i, line in enumerate(lines):
            if r'\toprule' in line:
                toprule_idx = i
                break
        
        if toprule_idx != -1:
            # Construct the two header lines
            header_line1 = "Metric & " + " & ".join(renamed_column_headers) + r" \\"
            # For the second header line, LLM in the first column, then empty cells for metrics
            header_line2 = "LLM & " + " & ".join([""] * len(renamed_column_headers)) + r" \\"
            
            custom_header_block = [
                header_line1,
                header_line2,
                r"\midrule" # Add \midrule after our custom header
            ]
            
            # Insert custom header block after \toprule
            lines = lines[:toprule_idx + 1] + custom_header_block + lines[toprule_idx + 1:]
        else:
            # This should not happen if to_latex includes \toprule, which it does by default
            # even with header=False, if booktabs=True (which is default).
            print(f"Warning: Could not find \\toprule to insert custom header for {caption}. Header might be missing or incorrect.")

        latex_string = '\n'.join(lines)
        
        print(f"Successfully generated LaTeX table snippet for {language.title()} {display_name_for_caption}.")
        return latex_string

    except Exception as e:
        print(f"Error generating LaTeX string for {language.title()} {display_name_for_caption}: {e}")
        print("DataFrame that was to be converted to LaTeX (df_for_latex):")
        print(df_for_latex.head())
        return None


def generate_latex_reports_by_method( # Consider renaming this function if its scope significantly broadens
    df_data: pd.DataFrame,
    output_dir: str,
    model_sort_order: list,
    metrics_to_include: list,
    config_data: dict
):
    """
    Generates .tex files, one for each specified retrieval method ("hybrid", "embedding", "keyword")
    and one for "Full Manual" (zeroshot at specified noise level).
    Each file contains tables for all configured languages for that method/report type.
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
    
    # Define all report types to generate
    # Each item is a tuple: (internal_id_or_algo_name, display_name_for_file, actual_filter_id_for_function)
    # For standard algorithms, internal_id and actual_filter_id are the same.
    # For Full Manual, internal_id is a unique key, actual_filter_id is FULL_MANUAL_INTERNAL_ID.
    
    report_configurations = []

    # Standard RAG algorithms
    standard_rag_methods = ["hybrid", "embedding", "keyword"]
    available_retrieval_methods_in_data = df_data['retrieval_algorithm'].str.lower().unique()
    
    for method in standard_rag_methods:
        if method in available_retrieval_methods_in_data:
            report_configurations.append(
                (method, method, method) # (id, filename_base, param_for_generate_single_table)
            )
        else:
            print(f"Info: Standard retrieval method '{method}' not found in data. Skipping its .tex file generation.")

    # Full Manual configuration
    # Check if data for Full Manual exists before adding it to configurations
    has_full_manual_data = not df_data[
        (df_data['retrieval_algorithm'].str.lower() == "zeroshot") &
        (df_data['noise_level'] == FULL_MANUAL_NOISE_LEVEL)
    ].empty
    
    if has_full_manual_data:
        report_configurations.append(
            ("full_manual", "full_manual", FULL_MANUAL_INTERNAL_ID) 
        )
        print(f"Info: Data for '{FULL_MANUAL_ALIAS}' (zeroshot at noise {FULL_MANUAL_NOISE_LEVEL}) found. Will generate its .tex file.")
    else:
        print(f"Info: No data found for '{FULL_MANUAL_ALIAS}' (zeroshot at noise {FULL_MANUAL_NOISE_LEVEL}). Skipping its .tex file generation.")


    if not report_configurations:
        print("Warning: No data found for any of the target report types. No .tex files will be generated.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for report_id, filename_base, generation_param in report_configurations:
        
        display_title = FULL_MANUAL_ALIAS if generation_param == FULL_MANUAL_INTERNAL_ID else report_id.title()
        print(f"\n--- Processing Report Type: {display_title} ---")
        
        all_tables_for_this_report_type_latex = []

        for lang in languages_from_config:
            table_latex_string = _generate_single_latex_table_string(
                df_data=df_data,
                language=lang,
                retrieval_algorithm_or_report_type=generation_param, # Pass the correct param here
                model_sort_order=model_sort_order,
                metrics_to_include=metrics_to_include
            )
            if table_latex_string:
                all_tables_for_this_report_type_latex.append(table_latex_string)
        
        if all_tables_for_this_report_type_latex:
            separator = "\n\n"
            combined_latex_content = separator.join(all_tables_for_this_report_type_latex)
            
            output_filename = f"{filename_base}_all_languages_performance.tex"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                with open(output_path, "w", encoding='utf-8') as f:
                    f.write(combined_latex_content)
                print(f"Successfully generated consolidated LaTeX file: {output_path}")
            except IOError as e:
                print(f"Error writing file {output_path}: {e}")
        else:
            print(f"No tables generated for report type: {display_title}. No .tex file created for it.")


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