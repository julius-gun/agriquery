import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from plot_config import (
    MODEL_PARAMETER_COUNTS, 
    clean_model_name,
    LANGUAGE_PALETTE
)

def get_model_size(model_raw_name: str) -> float:
    """Retrieves parameter count from config or attempts regex extraction."""
    if model_raw_name in MODEL_PARAMETER_COUNTS:
        return MODEL_PARAMETER_COUNTS[model_raw_name]
    
    # Fallback: Extract "7B", "1.5B" etc.
    match = re.search(r'(\d+(\.\d+)?)B', model_raw_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return 0.0

def generate_model_size_efficiency_scatter(df_data: pd.DataFrame, output_dir: str):
    """
    Generates a scatter plot: Model Size (Billions) vs F1 Score.
    Uses Markdown format as the baseline, averaged across all languages.
    """
    print("\n--- Generating Model Size Efficiency Scatter Plot ---")
    
    # Filter for Hybrid algorithm and Markdown extension
    df = df_data[
        (df_data['retrieval_algorithm'] == 'hybrid') & 
        (df_data['file_extension'] == 'md') &
        (df_data['metric_type'] == 'f1_score')
    ].copy()
    
    if df.empty:
        print("No suitable data found for efficiency plot.")
        return

    # Calculate average F1 score across all languages per model
    df_avg = df.groupby('question_model')['metric_value'].mean().reset_index()
    
    # Map sizes and clean names
    df_avg['size_b'] = df_avg['question_model'].apply(get_model_size)
    df_avg['display_name'] = df_avg['question_model'].apply(clean_model_name)
    
    # Sort by size for cleaner logic
    df_avg = df_avg.sort_values('size_b')
    df_avg = df_avg[df_avg['size_b'] > 0] # Filter out unknown sizes

    if df_avg.empty:
        print("No model sizes could be determined.")
        return

    # Plot
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # Scatter points
    sns.scatterplot(
        data=df_avg,
        x='size_b',
        y='metric_value',
        s=200,
        color="#2b7bba",
        edgecolor='w',
        linewidth=1.5,
        ax=ax
    )
    
    # Add Log Scale for X if range is large (1.5 to 120 is 2 orders of magnitude)
    ax.set_xscale('log')
    
    # Custom ticks for log scale to make it readable
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    
    # Direct Labeling
    texts = []
    for i, row in df_avg.iterrows():
        texts.append(ax.text(
            row['size_b'], 
            row['metric_value'], 
            row['display_name'],
            fontsize=10,
            weight='bold'
        ))
    
    # Avoid overlap
    adjust_text(
        texts, 
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_points=(1.5, 1.5)
    )

    ax.set_xlabel("Model Parameters (Billions) - Log Scale", fontsize=12)
    ax.set_ylabel("Average F1 Score (Markdown)", fontsize=12)
    ax.set_title("Model Efficiency: Parameter Count vs. Performance", fontsize=16, pad=15)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    
    # Save
    output_path = os.path.join(output_dir, "scatter_size_vs_f1.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def generate_performance_gap_plot(df_data: pd.DataFrame, output_dir: str):
    """
    Generates a Line Plot contrasting English vs German performance (Markdown).
    X-Axis: Models sorted by size.
    Y-Axis: F1 Score.
    Visualizes the "Cross-Lingual Tax".
    """
    print("\n--- Generating Performance Gap (Cross-Lingual Tax) Plot ---")
    
    target_langs = ['english', 'german']
    
    # Filter Data
    df = df_data[
        (df_data['retrieval_algorithm'] == 'hybrid') & 
        (df_data['file_extension'] == 'md') &
        (df_data['metric_type'] == 'f1_score') &
        (df_data['language'].isin(target_langs))
    ].copy()
    
    if df.empty:
        print("No English/German Markdown data found.")
        return
        
    # Pivot to get columns: model, english, german
    pivot = df.pivot_table(
        index='question_model', 
        columns='language', 
        values='metric_value'
    ).reset_index()
    
    # Check if we have both columns
    if 'english' not in pivot.columns or 'german' not in pivot.columns:
        print("Missing English or German data for gap plot.")
        return
        
    # Add size info for sorting
    pivot['size_b'] = pivot['question_model'].apply(get_model_size)
    pivot['display_name'] = pivot['question_model'].apply(clean_model_name)
    
    # Sort by size
    pivot = pivot.sort_values('size_b')
    
    # Prepare Plot
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    x_indices = range(len(pivot))
    
    # Plot Lines
    plt.plot(x_indices, pivot['english'], marker='o', linewidth=3, markersize=8, 
             color=LANGUAGE_PALETTE['english'], label='English (Ideal)')
             
    plt.plot(x_indices, pivot['german'], marker='s', linewidth=3, markersize=8, 
             color=LANGUAGE_PALETTE['german'], label='German (Harder Case)')
             
    # Fill the gap
    plt.fill_between(x_indices, pivot['english'], pivot['german'], 
                     color='gray', alpha=0.1, label='Cross-Lingual Tax')

    # Direct Labeling (At the end of the lines)
    last_idx = len(pivot) - 1
    last_en = pivot.iloc[last_idx]['english']
    last_de = pivot.iloc[last_idx]['german']
    
    ax.text(last_idx + 0.1, last_en, "English (Ideal)", 
            color=LANGUAGE_PALETTE['english'], va='center', fontweight='bold', fontsize=12)
    ax.text(last_idx + 0.1, last_de, "German (Harder)", 
            color=LANGUAGE_PALETTE['german'], va='center', fontweight='bold', fontsize=12)

    # Label the Gap on a specific model (e.g. Llama 3.2 3B if present, or just the biggest gap)
    # Finding Llama 3.2 3B
    gap_model_row = pivot[pivot['question_model'].str.contains("llama3.2_3B", case=False)]
    if not gap_model_row.empty:
        idx = pivot.index.get_loc(gap_model_row.index[0])
        val_en = gap_model_row.iloc[0]['english']
        val_de = gap_model_row.iloc[0]['german']
        mid_point = (val_en + val_de) / 2
        
        # Draw arrow/text for specific highlight
        ax.annotate(
            "Huge Gap\n(Llama 3.2 3B)", 
            xy=(idx, mid_point), 
            xytext=(idx, mid_point - 0.15),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            ha='center', fontsize=10
        )

    # X-Axis settings
    plt.xticks(x_indices, pivot['display_name'], rotation=45, ha='right')
    
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("The Cross-Lingual Tax: English vs. German Performance (Markdown)", fontsize=16, pad=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines for elegance
    sns.despine()

    # Save
    output_path = os.path.join(output_dir, "line_performance_gap_english_german.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()