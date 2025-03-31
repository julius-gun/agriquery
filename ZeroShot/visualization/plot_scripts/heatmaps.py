# python -m visualization.plot_scripts.heatmaps
# visualization/plot_scripts/heatmaps.py
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, List

def create_heatmap_metric_per_files(visualization_data: Dict, output_dir: str, metric: str):
    """
    Creates a heatmap showing a specified metric for each file and model combination.
    Averages across noise levels, languages, and datasets.
    Direct labels are used to display metric scores on the heatmap.
    Rows: Models
    Columns: File Formats

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The metric to display (e.g., 'f1_score', 'accuracy', 'precision', 'recall').
    """
    file_extensions, model_names, metric_matrix = _prepare_heatmap_data_files(visualization_data, metric)
    if not file_extensions or not model_names:
        print(f"No data to plot heatmap for metric: {metric}.")
        return

    _plot_and_save_heatmap_files(output_dir, metric, file_extensions, model_names, metric_matrix)


def _prepare_heatmap_data_files(visualization_data, metric):
    """Prepares data for heatmap of metric per files."""
    file_extensions = sorted(list(set(ext for model_data in visualization_data.values() for ext in model_data.keys())))
    model_names = sorted(list(visualization_data.keys()))

    metric_matrix = np.zeros((len(model_names), len(file_extensions)))

    # Populate the matrix with metric scores
    for i, model_name in enumerate(model_names):
        for j, file_extension in enumerate(file_extensions):
            file_data = visualization_data.get(model_name, {}).get(file_extension, {})
            all_noise_levels_metric = []
            for dataset_name, dataset_data in file_data.items(): # Iterate through datasets
                for lang_data in dataset_data.values(): # Iterate through languages
                    for noise_level_metrics in lang_data.values(): # Iterate through noise levels
                        all_noise_levels_metric.append(noise_level_metrics.get(metric, np.nan))
            avg_metric = np.nanmean(all_noise_levels_metric) if all_noise_levels_metric else np.nan
            metric_matrix[i, j] = avg_metric
    return file_extensions, model_names, metric_matrix


def _plot_and_save_heatmap_files(output_dir, metric, file_extensions, model_names, metric_matrix):
    """Plots and saves the heatmap for metric per files."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(metric_matrix, cmap="viridis", aspect="auto")

    _set_heatmap_labels_and_ticks(ax, file_extensions, model_names, metric)
    _add_colorbar(fig, ax, im, metric)
    _add_cell_labels(ax, metric_matrix)
    _add_heatmap_title(ax, metric, "Metric per Files", noise_levels=["All"], averaged_across="Noise Levels, Languages, Datasets", im=im, fig=fig) # ADDED: Call to _add_heatmap_title

    fig.tight_layout()
    _save_heatmap_image(output_dir, metric, "files", fig) # noise_levels is not passed here


def create_heatmap_llm_vs_file_format(
    visualization_data: Dict, output_dir: str, metric: str, noise_levels: List[int]
):
    """
    Creates a heatmap of Average F1-Score: LLM Models vs. File Formats.
    Averages across languages for specified noise levels.
    Rows: Models, Columns: File Formats, Cell Color: Average Metric.
    Direct labels are used to display metric scores on the heatmap.

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The metric to display (e.g., 'f1_score', 'accuracy', 'precision', 'recall').
        noise_levels (List[int]): Noise levels to filter and average over.
    """
    file_extensions, model_names, metric_matrix = _prepare_heatmap_data_llm_file_format(visualization_data, metric, noise_levels)
    if not file_extensions or not model_names:
        print(f"No data to plot heatmap for metric: {metric} and noise levels: {noise_levels}.")
        return

    _plot_and_save_heatmap_llm_file_format(output_dir, metric, noise_levels, file_extensions, model_names, metric_matrix)


def _prepare_heatmap_data_llm_file_format(visualization_data, metric, noise_levels):
    """Prepares data for heatmap of LLM vs file format."""
    file_extensions = sorted(list(set(ext for model_data in visualization_data.values() for ext in model_data.keys())))
    model_names = sorted(list(visualization_data.keys()))
    metric_matrix = np.zeros((len(model_names), len(file_extensions)))

    for i, model_name in enumerate(model_names):
        for j, file_extension in enumerate(file_extensions):
            file_data = visualization_data.get(model_name, {}).get(file_extension, {})
            all_languages_noise_levels_metric = []
            for dataset_name, dataset_data in file_data.items(): # Iterate over datasets
                for lang_data in dataset_data.values(): # Iterate over languages
                    for noise_level in noise_levels:
                        if noise_level in lang_data:
                            all_languages_noise_levels_metric.append(
                                lang_data[noise_level].get(metric, np.nan)
                            )
            avg_metric = (
                np.nanmean(all_languages_noise_levels_metric)
                if all_languages_noise_levels_metric
                else np.nan
            )
            metric_matrix[i, j] = avg_metric
    return file_extensions, model_names, metric_matrix


def _plot_and_save_heatmap_llm_file_format(output_dir, metric, noise_levels, file_extensions, model_names, metric_matrix):
    """Plots and saves the heatmap for LLM vs file format."""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(metric_matrix, cmap="viridis", aspect="auto")

    _set_heatmap_labels_and_ticks(ax, file_extensions, model_names, metric, x_label="File Formats", y_label="LLM Models")
    _add_colorbar(fig, ax, im, metric)
    _add_cell_labels(ax, metric_matrix)
    _add_heatmap_title(ax, metric, "LLM Models vs File Formats", noise_levels, "Languages", im, fig) # pass fig and im here

    fig.tight_layout()
    noise_level_str_filename = "_".join(map(str, noise_levels))
    _save_heatmap_image(output_dir, metric, f"llm_vs_file_format_noise_{noise_level_str_filename}", fig, noise_levels=noise_levels) # noise_levels is passed here


def create_heatmap_llm_vs_language(
    visualization_data: Dict, output_dir: str, metric: str, noise_levels: List[int]
):
    """
    Creates a heatmap of Average Metric: LLM Models vs. Language.
    Averages across file formats for specified noise levels.
    Rows: Models, Columns: Languages, Cell Color: Average Metric.
    Direct labels are used to display metric scores on the heatmap.

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The metric to display (e.g., 'f1_score', 'accuracy', 'precision', 'recall').
        noise_levels (List[int]): Noise levels to filter and average over.
    """
    languages, model_names, metric_matrix = _prepare_heatmap_data_llm_language(visualization_data, metric, noise_levels)
    if not languages or not model_names:
        print(f"No data to plot heatmap for metric: {metric} and noise levels: {noise_levels}.")
        return

    _plot_and_save_heatmap_llm_language(output_dir, metric, noise_levels, languages, model_names, metric_matrix)


def _prepare_heatmap_data_llm_language(visualization_data, metric, noise_levels):
    """Prepares data for heatmap of LLM vs language."""
    languages = sorted(list(set(language for model_data in visualization_data.values() for file_extension_data in model_data.values() for dataset_data in file_extension_data.values() for language in dataset_data.keys())))
    model_names = sorted(list(visualization_data.keys()))
    metric_matrix = np.zeros((len(model_names), len(languages)))

    for i, model_name in enumerate(model_names):
        for j, language in enumerate(languages):
            model_data = visualization_data.get(model_name, {})
            all_files_noise_levels_metric = []
            for file_extension_data in model_data.values():
                for dataset_name, dataset_data in file_extension_data.items(): # Iterate over datasets
                    if language in dataset_data:
                        lang_data = dataset_data[language]
                        for noise_level in noise_levels:
                            if noise_level in lang_data:
                                all_files_noise_levels_metric.append(
                                    lang_data[noise_level].get(metric, np.nan)
                                )
            avg_metric = (
                np.nanmean(all_files_noise_levels_metric)
                if all_files_noise_levels_metric
                else np.nan
            )
            metric_matrix[i, j] = avg_metric
    return languages, model_names, metric_matrix


def _plot_and_save_heatmap_llm_language(output_dir, metric, noise_levels, languages, model_names, metric_matrix):
    """Plots and saves the heatmap for LLM vs language."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(metric_matrix, cmap="viridis", aspect="auto")

    _set_heatmap_labels_and_ticks(ax, languages, model_names, metric, x_label="Languages", y_label="LLM Models")
    _add_colorbar(fig, ax, im, metric)
    _add_cell_labels(ax, metric_matrix)
    _add_heatmap_title(ax, metric, "LLM Models vs Languages", noise_levels, "File Formats", im, fig) # pass fig and im here

    fig.tight_layout()
    noise_level_str_filename = "_".join(map(str, noise_levels))
    _save_heatmap_image(output_dir, metric, f"llm_vs_language_noise_{noise_level_str_filename}", fig, noise_levels=noise_levels) # noise_levels is passed here


def _set_heatmap_labels_and_ticks(ax, x_tick_labels, y_tick_labels, metric_name, x_label=None, y_label=None):
    """Sets common labels and ticks for heatmaps."""
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def _add_colorbar(fig, ax, im, metric_name):
    """Adds a colorbar to the heatmap."""
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"Average {metric_name.replace('_', ' ').title()}", rotation=-90, va="bottom")


def _add_cell_labels(ax, metric_matrix):
    """Adds direct labels to each cell of the heatmap."""
    for i in range(metric_matrix.shape[0]):
        for j in range(metric_matrix.shape[1]):
            text_color = 'w' if metric_matrix[i, j] < np.nanmean(metric_matrix) else 'black'
            ax.text(j, i, f"{metric_matrix[i, j]:.2f}", ha="center", va="center", color=text_color)


def _add_heatmap_title(ax, metric_name, chart_type, noise_levels, averaged_across, im, fig): # fig and im are now arguments
    """Adds a title to the heatmap."""
    title_noise_levels = ", ".join(map(str, noise_levels))
    ax.set_title(
        f"Average {metric_name.replace('_', ' ').title()} Heatmap: {chart_type}\n(Noise Levels: {title_noise_levels}, Averaged across {averaged_across})"
    )

    fig.tight_layout()

def _save_heatmap_image(output_dir, metric_name, chart_type, fig, noise_levels=None): # noise_levels is now optional
    """Saves the heatmap image to a file."""
    plot_folder = os.path.join(output_dir, "visualization", "plots", "heatmap")
    os.makedirs(plot_folder, exist_ok=True)
    if noise_levels: # Check if noise_levels is provided before using it
        noise_level_str_filename = "_".join(map(str, noise_levels)) # Format noise levels for filename
        filepath = os.path.join(
            plot_folder, f"heatmap_{metric_name}_llm_vs_language_noise_{noise_level_str_filename}.png"
        )
    else:
        filepath = os.path.join(
            plot_folder, f"heatmap_{metric_name}_{chart_type}.png" # Filename without noise levels
        )

    plt.savefig(filepath)
    print(f"{metric_name.replace('_', ' ').title()} heatmap saved to {filepath}")
    plt.close(fig)



if __name__ == '__main__':
    # Example Usage (assuming visualization_data is loaded or available)
    results_directory = "results"  # Replace with your actual results directory
    from visualization.visualization_data_extractor import extract_metrics_from_results
    visualization_data = extract_metrics_from_results(results_directory)

    if visualization_data:
        metrics_to_plot = ["f1_score", "accuracy", "precision", "recall"]
        noise_levels_heatmap = [1000, 10000, 30000] # Noise levels for the new heatmaps

        for metric in metrics_to_plot:
            create_heatmap_metric_per_files(
                visualization_data, results_directory, metric
            )
            create_heatmap_llm_vs_file_format(
                visualization_data, results_directory, metric, noise_levels_heatmap
            )
            create_heatmap_llm_vs_language(
                visualization_data, results_directory, metric, noise_levels_heatmap
            )

    else:
        print(
            "No visualization data found. Please ensure results are processed and visualization_data is available."
        )