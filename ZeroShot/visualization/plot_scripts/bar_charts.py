# visualization/plot_scripts/bar_charts.py
# python -m visualization.plot_scripts.bar_charts
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict
from visualization.visualization_data_extractor import extract_metrics_from_results


def create_metric_comparison_bar_chart(
    visualization_data: Dict,
    output_dir: str,
    metric: str,
    noise_level: int = 1000,
    file_extension: str = "txt",
):
    """
    Creates a bar chart comparing a specified metric across different models for different languages.
    Averages across datasets. Direct labels are used instead of legends.

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The metric to compare (e.g., 'f1_score', 'accuracy', 'precision', 'recall').
        noise_level (int): The noise level to filter the data (default: 1000).
        file_extension (str): The file extension to filter the data (default: "txt").
    """

    # Filter data for the specified noise level and file extension
    filtered_data = _filter_data_for_bar_chart(
        visualization_data, file_extension, noise_level, metric
    )
    averaged_filtered_data = _average_metric_across_datasets(filtered_data)

    models = list(averaged_filtered_data.keys())
    if not models:
        print(
            f"No data found for noise level {noise_level} and file extension {file_extension}."
        )
        return

    languages = sorted(
        list(
            set(
                lang
                for model_data in averaged_filtered_data.values()
                for lang in model_data.keys()
            )
        )
    )
    metric_scores = {
        lang: [averaged_filtered_data[model].get(lang, 0) for model in models]
        for lang in languages
    }

    _plot_metric_comparison_bar_chart(
        output_dir,
        metric,
        noise_level,
        file_extension,
        models,
        languages,
        metric_scores,
    )


def _filter_data_for_bar_chart(visualization_data, file_extension, noise_level, metric):
    """Filters visualization data for the specified file extension and noise level."""
    filtered_data = {}
    for model_name, model_data in visualization_data.items():
        if file_extension in model_data:
            filtered_data[model_name] = {}
            for dataset_name, dataset_data in model_data[file_extension].items():
                for language, language_data in dataset_data.items():
                    if noise_level in language_data:
                        if language not in filtered_data[model_name]:
                            filtered_data[model_name][language] = []
                        filtered_data[model_name][language].append(
                            language_data[noise_level][metric]
                        )
    return filtered_data


def _average_metric_across_datasets(filtered_data):
    """Averages metric scores across datasets for each language and model."""
    averaged_filtered_data = {}
    for model_name, model_lang_data in filtered_data.items():
        averaged_filtered_data[model_name] = {}
        for language, metric_values in model_lang_data.items():
            averaged_filtered_data[model_name][language] = (
                np.mean(metric_values) if metric_values else 0
            )
    return averaged_filtered_data


def _plot_metric_comparison_bar_chart(
    output_dir, metric, noise_level, file_extension, models, languages, metric_scores
):
    """Plots the bar chart for metric comparison across models and languages."""
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )  # Increased figure height for better label spacing

    bars_container = []
    for i, lang in enumerate(languages):
        bars = ax.bar(x + i * width, metric_scores[lang], width, label=lang)
        bars_container.append(bars)

    _add_bar_labels(ax, bars_container, languages)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Models")
    ax.set_title(
        f"{metric.replace('_', ' ').title()} Comparison Across Models for Different Languages (Noise Level {noise_level}, {file_extension.upper()} Files, Averaged across Datasets)"
    )
    ax.set_xticks(x + width * (len(languages) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    fig.tight_layout()

    _save_bar_chart(output_dir, metric, noise_level, file_extension, fig)


def _add_bar_labels(ax, bars_container, languages):
    """Adds direct labels to the bars in the bar chart."""
    for bars, lang in zip(bars_container, languages):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                label_text = f"{lang}\n{height:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
                )


def _save_bar_chart(output_dir, metric, noise_level, file_extension, fig):
    """Saves the generated bar chart to a file."""
    plot_folder = os.path.join(output_dir, "visualization", "plots", "bar_chart")
    os.makedirs(plot_folder, exist_ok=True)
    filepath = os.path.join(
        plot_folder, f"{metric}_comparison_{noise_level}_{file_extension}.png"
    )
    plt.savefig(filepath, bbox_inches="tight")  # Ensure labels are not cut off
    print(
        f"{metric.replace('_', ' ').title()} comparison bar chart saved to {filepath}"
    )
    plt.close(fig)


def create_metric_comparison_bar_chart_file_formats(
    visualization_data: Dict, output_dir: str, metric: str, noise_level: int = 1000
):
    """
     Creates a bar chart comparing a specified metric across different file formats.
     Averages across languages and datasets. Direct labels are used instead of legends.
     X-axis: File formats, Y-axis: Metric.

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The metric to compare (e.g., 'f1_score', 'accuracy', 'precision', 'recall').
        noise_level (int): The noise level to filter the data (default: 1000).
    """
    filtered_data_by_noise = _filter_data_for_file_format_chart(
        visualization_data, noise_level, metric
    )
    file_format_metric_scores = _aggregate_metric_by_file_format(
        filtered_data_by_noise, metric
    )
    average_metric_scores = _average_scores_across_models(file_format_metric_scores)

    file_formats = list(average_metric_scores.keys())
    metric_values = list(average_metric_scores.values())

    if not file_formats:
        print(
            f"Warning: No data found for noise level {noise_level} to create file format comparison bar chart."
        )
        return

    _plot_file_format_bar_chart(
        output_dir, metric, noise_level, file_formats, metric_values
    )


def _filter_data_for_file_format_chart(visualization_data, noise_level, metric):
    """Filters data for file format comparison bar chart for a specific noise level."""
    filtered_data_by_noise = {}
    for model_name, model_data in visualization_data.items():
        filtered_data_by_noise[model_name] = {}
        for file_extension, file_extension_data in model_data.items():
            filtered_data_by_noise[model_name][
                file_extension
            ] = []  # Initialize as list, corrected from {}
            for (
                dataset_name,
                dataset_data,
            ) in file_extension_data.items():  # Iterate over datasets
                for language, language_data in dataset_data.items():
                    if noise_level in language_data:
                        if file_extension not in filtered_data_by_noise[model_name]:
                            filtered_data_by_noise[model_name][
                                file_extension
                            ] = []  # Initialize list to store metric values across languages and datasets
                        filtered_data_by_noise[model_name][file_extension].append(
                            language_data[noise_level][metric]
                        )
    return filtered_data_by_noise


def _aggregate_metric_by_file_format(filtered_data_by_noise, metric):
    """Aggregates metric scores by file format, across languages and datasets."""
    file_format_metric_scores = {}
    for model_name, model_file_data in filtered_data_by_noise.items():
        for (
            file_extension,
            metric_values,
        ) in (
            model_file_data.items()
        ):  # Now metric_values is a list of values across languages and datasets
            if file_extension not in file_format_metric_scores:
                file_format_metric_scores[file_extension] = []
            file_format_metric_scores[file_extension].extend(
                metric_values
            )  # Extend the list with metric values
    return file_format_metric_scores


def _average_scores_across_models(file_format_metric_scores):
    """Averages metric scores across models for each file format."""
    # Calculate average metric scores for each file format
    average_metric_scores = {
        file_extension: np.mean(scores) if scores else 0
        for file_extension, scores in file_format_metric_scores.items()
    }
    return average_metric_scores
def _plot_file_format_bar_chart(output_dir, metric, noise_level, file_formats, metric_values):
    """Plots the bar chart for file format comparison."""
    x = np.arange(len(file_formats))
    width = 0.7  # Wider bars for better visibility

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, metric_values, width, color="skyblue")

    _add_file_format_bar_labels(ax, bars)

    ax.set_ylabel(f"Average {metric.replace('_', ' ').title()}")
    ax.set_xlabel("File Formats")
    ax.set_title(
        f"{metric.replace('_', ' ').title()} Comparison Across File Formats (Noise Level {noise_level}, Averaged across Languages and Datasets)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(file_formats)
    ax.set_ylim(0, max(metric_values) * 1.1)
    fig.tight_layout()

    _save_file_format_bar_chart(output_dir, metric, noise_level, fig)


def _add_file_format_bar_labels(ax, bars):
    """Adds direct labels to the bars in the file format bar chart."""
    for bar in bars:
        yval = bar.get_height()
        label_text = f"{yval:.2f}"  # Just the value for file format chart
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            label_text,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
        )


def _save_file_format_bar_chart(output_dir, metric, noise_level, fig):
    """Saves the file format comparison bar chart to a file."""
    plot_folder = os.path.join(output_dir, "visualization", "plots", "bar_chart")
    os.makedirs(plot_folder, exist_ok=True)
    filepath = os.path.join(
        plot_folder, f"{metric}_comparison_file_formats_{noise_level}.png"
    )
    plt.savefig(filepath, bbox_inches='tight') # Ensure labels are not cut off
    print(
        f"{metric.replace('_', ' ').title()} comparison bar chart across file formats saved to {filepath}"
    )
    plt.close(fig)


def create_metric_comparison_bar_chart_datasets(
    visualization_data: Dict, output_dir: str, metric: str, noise_level: int, file_extension: str
):
    """
    Creates a bar chart comparing a specified metric across different datasets for a given model and file extension.
    Averages across languages. Direct labels are used instead of legends.
    X-axis: Datasets, Y-axis: Metric.

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The metric to compare (e.g., 'f1_score', 'accuracy', 'precision', 'recall').
        noise_level (int): The noise level to filter the data.
        file_extension (str): The file extension to filter the data.
    """
    filtered_data = _filter_data_for_dataset_chart(visualization_data, file_extension, noise_level, metric)
    dataset_metric_scores = _aggregate_metric_by_dataset(filtered_data, metric)
    average_metric_scores = _average_metric_across_languages(dataset_metric_scores)

    datasets = list(average_metric_scores.keys())
    metric_values = list(average_metric_scores.values())

    if not datasets:
        print(f"No data found for noise level {noise_level} and file extension {file_extension} to create dataset comparison bar chart.")
        return

    _plot_metric_comparison_bar_chart_datasets(output_dir, metric, noise_level, file_extension, datasets, metric_values)


def _filter_data_for_dataset_chart(visualization_data, file_extension, noise_level, metric):
    """Filters data for dataset comparison bar chart for a specific file extension and noise level."""
    filtered_data = {}
    for model_name, model_data in visualization_data.items():
        if file_extension in model_data:
            filtered_data[model_name] = {}
            if file_extension in model_data:
                filtered_data[model_name] = {}
                for dataset_name, dataset_data in model_data[file_extension].items():
                    filtered_data[model_name][dataset_name] = []
                    for language, language_data in dataset_data.items():
                        if noise_level in language_data:
                            if dataset_name not in filtered_data[model_name]:
                                filtered_data[model_name][dataset_name] = []
                            filtered_data[model_name][dataset_name].append(language_data[noise_level][metric])
    return filtered_data


def _aggregate_metric_by_dataset(filtered_data, metric):
    """Aggregates metric scores by dataset, across languages."""
    dataset_metric_scores = {}
    for model_name, model_dataset_data in filtered_data.items():
        for dataset_name, metric_values in model_dataset_data.items():
            if dataset_name not in dataset_metric_scores:
                dataset_metric_scores[dataset_name] = []
            dataset_metric_scores[dataset_name].extend(metric_values)
    return dataset_metric_scores


def _average_metric_across_languages(dataset_metric_scores):
    """Averages metric scores across languages for each dataset."""
    average_metric_scores = {
        dataset_name: np.mean(scores) if scores else 0
        for dataset_name, scores in dataset_metric_scores.items()
    }
    return average_metric_scores


def _plot_metric_comparison_bar_chart_datasets(output_dir, metric, noise_level, file_extension, datasets, metric_values):
    """Plots the bar chart for dataset comparison."""
    x = np.arange(len(datasets))
    width = 0.7

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, metric_values, width, color="skyblue")

    _add_dataset_bar_labels(ax, bars)

    ax.set_ylabel(f"Average {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Datasets")
    ax.set_title(
        f"{metric.replace('_', ' ').title()} Comparison Across Datasets (Noise Level {noise_level}, {file_extension.upper()} Files, Averaged across Languages)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylim(0, max(metric_values) * 1.1)
    fig.tight_layout()

    _save_dataset_bar_chart(output_dir, metric, noise_level, file_extension, fig)


def _add_dataset_bar_labels(ax, bars):
    """Adds direct labels to the bars in the dataset comparison bar chart."""
    for bar in bars:
        yval = bar.get_height()
        label_text = f"{yval:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            label_text,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
        )


def _save_dataset_bar_chart(output_dir, metric, noise_level, file_extension, fig):
    """Saves the dataset comparison bar chart to a file."""
    plot_folder = os.path.join(output_dir, "visualization", "plots", "bar_chart")
    os.makedirs(plot_folder, exist_ok=True)
    filepath = os.path.join(
        plot_folder, f"{metric}_comparison_datasets_{noise_level}_{file_extension}.png"
    )
    plt.savefig(filepath, bbox_inches='tight') # Ensure labels are not cut off
    print(
        f"{metric.replace('_', ' ').title()} comparison bar chart across datasets saved to {filepath}"
    )
    plt.close(fig)


if __name__ == "__main__":
    # Example usage (assuming you have visualization_data.json in your results directory)
    results_directory = "results"  # Replace with your actual results directory
    visualization_data = extract_metrics_from_results(results_directory)

    if visualization_data:
        metrics_to_plot = ["f1_score", "accuracy", "precision", "recall"]
        file_extensions_bar_charts = ["txt"]
        noise_levels_bar_charts = [1000, 5000]

        for metric in metrics_to_plot:
            for noise_level in noise_levels_bar_charts:
                for file_extension in file_extensions_bar_charts:
                    create_metric_comparison_bar_chart(
                        visualization_data,
                        results_directory,
                        metric,
                        noise_level=noise_level,
                        file_extension=file_extension,
                    )
                    create_metric_comparison_bar_chart_file_formats(
                        visualization_data,
                        results_directory,
                        metric,
                        noise_level=noise_level,
                    )
                    create_metric_comparison_bar_chart_datasets(
                        visualization_data,
                        results_directory,
                        metric,
                        noise_level=noise_level,
                        file_extension=file_extension
                    )

    else:
        print(
            "No visualization data found.  Run tests or ensure results are in the correct directory."
        )
