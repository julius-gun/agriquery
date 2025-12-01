import os
import pandas as pd
from typing import List, Optional

from plot_config import clean_model_name, METRIC_DISPLAY_NAMES

def _generate_single_language_table(
    df: pd.DataFrame,
    language: str,
    metrics: List[str],
    model_sort_order: Optional[List[str]]
) -> Optional[str]:
    """Generates a LaTeX tabular fragment for a specific language."""
    
    # Filter for Language
    df_lang = df[df['language'] == language].copy()
    if df_lang.empty: return None

    # Pivot: Index=Model, Columns=Metric
    pivot = df_lang.pivot_table(
        index='question_model', 
        columns='metric_type', 
        values='metric_value',
        aggfunc='mean' # Handle potential duplicates by averaging
    )
    
    # Sort Models
    if model_sort_order:
        # Create mapping of cleaned names to sort order
        cleaned_order = [clean_model_name(m) for m in model_sort_order]
        # Filter to models actually present
        present = [m for m in cleaned_order if m in pivot.index]
        # Add any remaining models sorted alphabetically
        remaining = sorted([m for m in pivot.index if m not in present])
        pivot = pivot.reindex(present + remaining)
    
    # Select and Order Metrics
    valid_metrics = [m for m in metrics if m in pivot.columns]
    if not valid_metrics: return None
    pivot = pivot[valid_metrics]

    # Format Data: Bold Max, 3 decimal places
    df_fmt = pivot.copy()
    for col in df_fmt.columns:
        max_val = df_fmt[col].max()
        df_fmt[col] = df_fmt[col].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == max_val else f"{x:.3f}"
        )

    # Rename Columns
    df_fmt.columns = [METRIC_DISPLAY_NAMES.get(c, c.title()) for c in df_fmt.columns]
    
    # Escape underscores in index (Model Names)
    df_fmt.index = df_fmt.index.str.replace("_", r"\_")
    df_fmt.index.name = None # Remove index name for cleaner LaTeX

    # Generate LaTeX
    # Note: Caption/Label are placeholders here; usually managed by caller or specific logic
    latex = df_fmt.to_latex(
        column_format="l" + "c" * len(df_fmt.columns),
        caption=f"Hybrid RAG Performance: {language.title()}",
        label=f"tab:hybrid_{language}",
        escape=False, # Allow \textbf
        position="!htbp"
    )
    
    return latex

def generate_latex_report(
    df_data: pd.DataFrame,
    output_dir: str,
    model_sort_order: Optional[List[str]] = None
):
    """
    Generates consolidated .tex files containing tables for all languages.
    Generates one file per file format (md, xml, json).
    """
    print("\n--- Generating LaTeX Tables (Hybrid / per Format) ---")
    
    if df_data is None or df_data.empty:
        print("No data provided.")
        return

    # Filter: Hybrid
    df_hybrid = df_data[df_data['retrieval_algorithm'] == 'hybrid'].copy()

    if df_hybrid.empty:
        print("No Hybrid data found.")
        return
        
    extensions = sorted(df_hybrid['file_extension'].dropna().unique())
    ext_map = {'md': 'Markdown', 'json': 'JSON', 'xml': 'XML'}

    for ext in extensions:
        df_filtered = df_hybrid[df_hybrid['file_extension'] == ext].copy()
        if df_filtered.empty: continue

        # Clean Model Names
        df_filtered['question_model'] = df_filtered['question_model'].apply(clean_model_name)

        languages = sorted(df_filtered['language'].unique())
        metrics = ['f1_score', 'accuracy', 'precision', 'recall']
        
        all_tables = []
        ext_display = ext_map.get(ext, ext.upper())
        
        # Add file header
        all_tables.append(f"% Tables for Format: {ext_display}")

        for lang in languages:
            table_str = _generate_single_language_table(df_filtered, lang, metrics, model_sort_order)
            if table_str:
                # Modify Caption and Label to include format
                table_str = table_str.replace(f"tab:hybrid_{lang}", f"tab:hybrid_{lang}_{ext}")
                table_str = table_str.replace(f"Performance: {lang.title()}", f"Performance: {lang.title()} ({ext_display})")
                all_tables.append(table_str)

        if all_tables:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"tables_hybrid_{ext}.tex")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_tables))
                
            print(f"Generated LaTeX report: {output_path}")
