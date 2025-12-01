from typing import List, Optional, Tuple, Dict
import pandas as pd
from .barcharts import create_grouped_barchart

def create_english_retrieval_barchart(
    data: pd.DataFrame,
    output_path: str,
    metric_name: str,
    model_sort_order: Optional[List[str]] = None,
    retrieval_method_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Wrapper for creating English specific comparison charts using the generic barchart function.
    """
    # Define a specific palette for Retrieval Methods if desired, or let seaborn handle it
    # For now, we use a default high-contrast set or passed palette
    method_palette = {
        "Hybrid": "#1f77b4",     # Blue
        "Embedding": "#ff7f0e",  # Orange
        "Keyword": "#2ca02c"     # Green
    }

    create_grouped_barchart(
        data=data,
        output_path=output_path,
        metric_name=metric_name,
        x_col="question_model",
        y_col="metric_value",
        hue_col="retrieval_method_display",
        x_order=model_sort_order,
        hue_order=retrieval_method_order,
        palette=method_palette,
        title=f"English Retrieval Method Comparison ({metric_name})",
        xlabel="LLM Model",
        figsize=figsize
    )
