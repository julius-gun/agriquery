import os
import sys
import argparse
import pandas as pd
from typing import List, Optional, Dict, Tuple

try:
    from .plot_utils import add_project_paths, sanitize_filename
    from .english_barcharts import create_english_retrieval_barchart
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import add_project_paths, sanitize_filename
    from english_barcharts import create_english_retrieval_barchart

add_project_paths()

from visualization_data_extractor import extract_detailed_visualization_data
from utils.config_loader import ConfigLoader

class EnglishBarchartConfig:
    TARGET_LANGUAGE: str = "english"
    TARGET_METRICS: List[str] = ["accuracy", "f1_score"]
    RETRIEVAL_METHODS: Dict[str, str] = {
        "hybrid": "Hybrid",
        "embedding": "Embedding",
        "keyword": "Keyword",
    }
    FULL_MANUAL_ALIAS: str = "Full Manual"
    FULL_MANUAL_NOISE_LEVEL: int = 59000
    DISPLAY_ORDER: List[str] = ["Hybrid", "Embedding", "Keyword", FULL_MANUAL_ALIAS]
    MODEL_MAPPINGS: Dict[str, str] = {
        "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
        "phi3_14B_q4_medium-128k": "phi3 14B",
    }

def clean_model_name(model_name: str) -> str:
    cleaned_name = EnglishBarchartConfig.MODEL_MAPPINGS.get(model_name, model_name)
    if cleaned_name == model_name:
        if cleaned_name.endswith("-128k"):
            cleaned_name = cleaned_name.removesuffix("-128k")
        cleaned_name = cleaned_name.replace("_", " ")
    return cleaned_name

def generate_english_retrieval_comparison_barcharts(
    df_data: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "",
    model_sort_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (18, 9)
) -> None:
    print("\n--- Generating English Retrieval Comparison Bar Charts ---")

    # Filter for Markdown only to ensure clean comparison of Algorithms
    if 'file_extension' in df_data.columns:
        df_filtered = df_data[df_data['file_extension'] == 'md'].copy()
        if df_filtered.empty:
            print("Warning: No Markdown files found for English comparison. Using all formats.")
            df_filtered = df_data.copy()
        else:
            print("Filtered data to 'md' extension for English retrieval comparison.")
    else:
        df_filtered = df_data.copy()

    df_plot_base = df_filtered.copy()
    df_plot_base['question_model'] = df_plot_base['question_model'].apply(clean_model_name)
    
    cleaned_model_sort_order = None
    if model_sort_order:
        cleaned_model_sort_order = [clean_model_name(m) for m in model_sort_order]

    # Filter English
    df_english = df_plot_base[df_plot_base["language"] == EnglishBarchartConfig.TARGET_LANGUAGE].copy()
    if df_english.empty:
        print("No English data found.")
        return

    # Prepare Full Manual
    df_full_manual = df_english[
        (df_english["retrieval_algorithm"] == "zeroshot") &
        (df_english["noise_level"] == EnglishBarchartConfig.FULL_MANUAL_NOISE_LEVEL)
    ].copy()
    if not df_full_manual.empty:
        df_full_manual["retrieval_method_display"] = EnglishBarchartConfig.FULL_MANUAL_ALIAS

    # Prepare RAG
    rag_frames = []
    for internal, display in EnglishBarchartConfig.RETRIEVAL_METHODS.items():
        subset = df_english[df_english["retrieval_algorithm"] == internal].copy()
        if not subset.empty:
            subset["retrieval_method_display"] = display
            rag_frames.append(subset)
            
    all_frames = rag_frames + ([df_full_manual] if not df_full_manual.empty else [])
    if not all_frames:
        return

    df_combined = pd.concat(all_frames, ignore_index=True)
    
    # Sort Orders
    actual_methods = df_combined['retrieval_method_display'].unique().tolist()
    final_method_order = [m for m in EnglishBarchartConfig.DISPLAY_ORDER if m in actual_methods]
    
    # Plot
    for metric in EnglishBarchartConfig.TARGET_METRICS:
        df_metric = df_combined[df_combined["metric_type"] == metric].copy()
        if df_metric.empty:
            continue
            
        filename = f"{output_filename_prefix}english_retrieval_{sanitize_filename(metric)}_comparison.png"
        output_filepath = os.path.join(output_dir, filename)
        
        create_english_retrieval_barchart(
            data=df_metric,
            output_path=output_filepath,
            metric_name=metric,
            model_sort_order=cleaned_model_sort_order,
            retrieval_method_order=final_method_order,
            figsize=figsize
        )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    visualization_dir = os.path.dirname(current_dir) 
    project_root_dir = os.path.dirname(visualization_dir) 

    default_config_path = os.path.join(project_root_dir, "config.json")
    default_results_path = os.path.join(project_root_dir, "results")
    default_output_path = os.path.join(visualization_dir, "plots", "english_retrieval_barcharts")

    parser = argparse.ArgumentParser(description="Generate English Retrieval Comparison Bar Charts.")
    parser.add_argument("--results-dir", type=str, default=default_results_path)
    parser.add_argument("--output-dir", type=str, default=default_output_path)
    parser.add_argument("--config-path", type=str, default=default_config_path)

    args = parser.parse_args()

    df_all_data = extract_detailed_visualization_data(args.results_dir)
    if df_all_data is None or df_all_data.empty:
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    standalone_model_sort_order = None
    try:
        config_loader = ConfigLoader(args.config_path)
        viz_settings = config_loader.config.get("visualization_settings", {})
        standalone_model_sort_order = viz_settings.get("REPORT_MODEL_SORT_ORDER")
    except:
        pass

    generate_english_retrieval_comparison_barcharts(
        df_data=df_all_data,
        output_dir=args.output_dir,
        model_sort_order=standalone_model_sort_order
    )
