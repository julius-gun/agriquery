# plot_existing_results.py
# python -m visualization.plot_existing_results
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.config_loader import ConfigLoader
from utils.metrics import calculate_metrics
from visualization.plotting import (
    create_duration_boxplots,
    create_accuracy_boxplots,
    create_precision_boxplots,
    create_recall_boxplots,
    create_f1_score_boxplots,
)


def load_results_from_file(filepath):
    """Loads results from a single JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading results from {filepath}: {e}")
        return None


def extract_data_for_plotting(
    results, language_from_filename=False, result_filename=None
):
    """
    Extracts relevant data from a list of result objects for plotting.
    Infers language from filename if language_from_filename is True.
    """
    plot_data = []
    for result in results:
        language = "unknown"  # default language
        dataset_type = "all"  # default dataset type
        if language_from_filename and result_filename:
            parts = result_filename.split("_")
            if parts:
                potential_language = parts[0]
                if potential_language in ["english", "german", "french"]:
                    language = potential_language
                if "pairs" in result_filename:
                    dataset_type = "pairs"
                elif "tables" in result_filename:
                    dataset_type = "tables"
                elif "unanswerable" in result_filename:
                    dataset_type = "unanswerable"

        plot_data.append(
            {
                "model_name": result["model_name"],
                "duration": result["duration"],
                "noise_level": result["noise_level"],
                "file_extension": result["file_extension"],
                "context_type": result["context_type"],
                "language": language,
                "dataset_type": dataset_type,  # Include dataset type
                "self_evaluation": result.get("self_evaluation", "N/A"),
                # --- NO metrics here ---
                "question": result["question"],  # Add question for grouping
                "expected_answer": result["expected_answer"],  # Add expected answer
                # Include the actual results for later metric calculation
                "true_positives": result.get("true_positives", 0),
                "true_negatives": result.get("true_negatives", 0),
                "false_positives": result.get("false_positives", 0),
                "false_negatives": result.get("false_negatives", 0),
            }
        )
    return plot_data


def create_combined_metric_boxplots(results_dir, output_dir):
    """
    Loads all result files, combines data, and creates box plots.
    """
    all_plot_data = []
    for filename in os.listdir(results_dir):  # filename is the result filename here
        if filename.endswith("_results.json"):
            filepath = os.path.join(results_dir, filename)
            results = load_results_from_file(filepath)
            if results:
                plot_data = extract_data_for_plotting(
                    results, language_from_filename=True, result_filename=filename
                )  # pass filename here
                if plot_data:
                    all_plot_data.extend(plot_data)

    if not all_plot_data:
        print("No data available to create combined box plots.")
        return

    df = pd.DataFrame(all_plot_data)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Group data BEFORE calculating metrics ---
    grouped_data = df.groupby(
        ["model_name", "noise_level", "file_extension", "context_type", "language"]
    )

    # --- Calculate metrics for each group ---
    aggregated_data = grouped_data.apply(
        lambda x: pd.Series(calculate_metrics(x.to_dict("records"))),
        include_groups=False,
    ).reset_index()
    # Merge duration (keep original values, don't average)
    duration_data = (
        df[
            [
                "model_name",
                "noise_level",
                "file_extension",
                "context_type",
                "language",
                "duration",
            ]
        ]
        .groupby(
            ["model_name", "noise_level", "file_extension", "context_type", "language"],
            as_index=False,
        )
        .mean()
    )  # take mean of duration
    aggregated_data = pd.merge(
        aggregated_data,
        duration_data,
        on=["model_name", "noise_level", "file_extension", "context_type", "language"],
        suffixes=("_metric", "_duration"),
    )  # added suffixes to distinguish columns
    aggregated_data = aggregated_data.rename(
        columns={"duration_duration": "duration"}
    )  # rename duration column to duration

    # --- Now create plots using the aggregated data ---
    print("Creating Duration Boxplots...")
    create_duration_boxplots(
        results=aggregated_data.copy(),
        output_dir=output_dir,
        hue_parameter="model_name",
    )
    create_duration_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="language"
    )
    create_duration_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="file_extension"
    )
    create_duration_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="context_type"
    )

    # --- Accuracy Boxplots ---
    print("Creating Accuracy Boxplots...")
    create_accuracy_boxplots(
        results=aggregated_data.copy(),
        output_dir=output_dir,
        hue_parameter="model_name",
    )
    create_accuracy_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="language"
    )
    create_accuracy_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="file_extension"
    )
    create_accuracy_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="context_type"
    )

    # --- Precision Boxplots ---
    print("Creating Precision Boxplots...")
    create_precision_boxplots(
        results=aggregated_data.copy(),
        output_dir=output_dir,
        hue_parameter="model_name",
    )
    create_precision_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="language"
    )
    create_precision_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="file_extension"
    )
    create_precision_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="context_type"
    )

    # --- Recall Boxplots ---
    print("Creating Recall Boxplots...")
    create_recall_boxplots(
        results=aggregated_data.copy(),
        output_dir=output_dir,
        hue_parameter="model_name",
    )
    create_recall_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="language"
    )
    create_recall_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="file_extension"
    )
    create_recall_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="context_type"
    )

    # --- F1 Score Boxplots ---
    print("Creating F1 Score Boxplots...")
    create_f1_score_boxplots(
        results=aggregated_data.copy(),
        output_dir=output_dir,
        hue_parameter="model_name",
    )
    create_f1_score_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="language"
    )
    create_f1_score_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="file_extension"
    )
    create_f1_score_boxplots(
        results=aggregated_data, output_dir=output_dir, hue_parameter="context_type"
    )

    # --- Dataset Specific Boxplots ---
    dataset_types = df["dataset_type"].unique()
    for dataset_type in dataset_types:
        dataset_df = df[df["dataset_type"] == dataset_type]
        if not dataset_df.empty:
            # Group by dataset type as well
            grouped_dataset_data = dataset_df.groupby(
                [
                    "model_name",
                    "noise_level",
                    "file_extension",
                    "context_type",
                    "language",
                    "dataset_type",
                ]
            )
            # Correctly calculate metrics within the apply function
            aggregated_dataset_data = grouped_dataset_data.apply(
                lambda x: pd.Series(calculate_metrics(x.to_dict("records"))),
                include_groups=False,
            ).reset_index()

            # Merge duration
            duration_dataset_data = (
                dataset_df[
                    [
                        "model_name",
                        "noise_level",
                        "file_extension",
                        "context_type",
                        "language",
                        "dataset_type",
                        "duration",
                    ]
                ]
                .groupby(
                    [
                        "model_name",
                        "noise_level",
                        "file_extension",
                        "context_type",
                        "language",
                        "dataset_type",
                    ],
                    as_index=False,
                )
                .mean()
            )  # take mean of duration
            aggregated_dataset_data = pd.merge(
                aggregated_dataset_data,
                duration_dataset_data,
                on=[
                    "model_name",
                    "noise_level",
                    "file_extension",
                    "context_type",
                    "language",
                    "dataset_type",
                ],
            )

            print(f"\nCreating Boxplots for Dataset: {dataset_type}...")
            create_accuracy_boxplots(
                results=aggregated_dataset_data.copy(),
                output_dir=output_dir,
                hue_parameter="model_name",
                dataset_type=dataset_type,
            )
            create_f1_score_boxplots(
                results=aggregated_dataset_data.copy(),
                output_dir=output_dir,
                hue_parameter="model_name",
                dataset_type=dataset_type,
            )
            create_precision_boxplots(
                results=aggregated_dataset_data.copy(),
                output_dir=output_dir,
                hue_parameter="model_name",
                dataset_type=dataset_type,
            )
            create_recall_boxplots(
                results=aggregated_dataset_data.copy(),
                output_dir=output_dir,
                hue_parameter="model_name",
                dataset_type=dataset_type,
            )


def main():
    config = ConfigLoader()
    results_dir = config.get_output_dir()  # Use the same results directory
    visualization_dir = config.config.get(
        "visualization_dir", "results/plots"
    )  # Define visualization dir in config

    create_combined_metric_boxplots(results_dir, visualization_dir)


if __name__ == "__main__":
    main()
