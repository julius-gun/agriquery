# python -m visualization.plot_scripts.box_plots
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import numpy as np

def create_duration_boxplot_file_formats(
    visualization_data: Dict, output_dir: str, noise_level: int = 1000
):
    """
    Creates a box plot comparing duration across different file formats.
    Averages across languages and datasets. Direct labels are used for median values.
    X-axis: File formats, Y-axis: Duration (seconds).

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        noise_level (int): The noise level to filter the data (default: 1000).
    """
    filtered_data_by_noise = _filter_duration_data(visualization_data, noise_level)
    file_format_duration_scores = _aggregate_duration_by_file_format(filtered_data_by_noise)
    file_formats = list(file_format_duration_scores.keys())
    duration_values = [file_format_duration_scores[file_format] for file_format in file_formats]

    if not file_formats:
        _handle_no_data_warning(noise_level)
        return

    _plot_duration_boxplot(output_dir, noise_level, file_formats, duration_values)


def _filter_duration_data(visualization_data, noise_level):
    """Filters visualization data for duration box plot for a specific noise level."""
    filtered_data_by_noise = {}
    for model_name, model_data in visualization_data.items():
        filtered_data_by_noise[model_name] = {}
        for file_extension, file_extension_data in model_data.items():
            filtered_data_by_noise[model_name][file_extension] = [] # Initialize as list, corrected from {}
            for dataset_name, dataset_data in file_extension_data.items(): # Iterate over datasets
                for language, language_data in dataset_data.items():
                    if noise_level in language_data:
                        filtered_data_by_noise[model_name][file_extension].append(language_data[noise_level]["duration_mean"])
    return filtered_data_by_noise


def _aggregate_duration_by_file_format(filtered_data_by_noise):
    """Aggregates duration scores by file format, averaging across languages and datasets."""
    file_format_duration_scores = {}
    for model_name, model_file_data in filtered_data_by_noise.items():
        for file_extension, duration_values in model_file_data.items(): # Now duration_values is a list across languages and datasets
            if file_extension not in file_format_duration_scores:
                file_format_duration_scores[file_extension] = []
            file_format_duration_scores[file_extension].extend(duration_values) # Extend with duration values
    return file_format_duration_scores
            

def _handle_no_data_warning(noise_level):
    """Prints a warning message when no data is found for the box plot."""
    print(f"Warning: No data found for noise level {noise_level} to create duration box plot.")


def _plot_duration_boxplot(output_dir, noise_level, file_formats, duration_values):
    """Plots the duration box plot and saves it to a file."""
    fig, ax = plt.subplots(figsize=(10, 6))
    boxes = ax.boxplot(duration_values, positions=np.arange(len(file_formats)), widths=0.6, patch_artist=True)

    _customize_boxplot_appearance(boxes)
    _add_boxplot_labels(ax, noise_level, file_formats, boxes)
    _save_boxplot(output_dir, noise_level, fig)


def _customize_boxplot_appearance(boxes):
    """Customizes the appearance of the box plot."""
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightseagreen', 'lightskyblue']
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)


def _add_boxplot_labels(ax, noise_level, file_formats, boxes):
    """Adds labels and title to the box plot, including direct labels for medians."""
    ax.set_ylabel('Duration (seconds)')
    ax.set_xlabel('File Formats')
    ax.set_title(f'Duration Comparison Across File Formats (Noise Level {noise_level}, Averaged across Languages and Datasets)')
    ax.set_xticks(np.arange(len(file_formats)))
    ax.set_xticklabels(file_formats)

    # Add direct labels for median values
    for median_line in boxes['medians']:
        x, y = median_line.get_xydata()[1] # x-coordinate, median value
        ax.text(x, y, f'{y:.2f}s', ha='left', va='center', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)) # Added background for readability


def _save_boxplot(output_dir, noise_level, fig):
    """Saves the duration box plot to a file."""
    plot_folder = os.path.join(output_dir, "visualization", "plots", "box_plot")
    os.makedirs(plot_folder, exist_ok=True)
    filepath = os.path.join(plot_folder, f"duration_boxplot_file_formats_{noise_level}.png")
    plt.savefig(filepath, bbox_inches='tight') # Ensure labels are not cut off
    print(f"Duration box plot across file formats saved to {filepath}")
    plt.close(fig)


if __name__ == '__main__':
    # Example usage (assuming you have visualization_data.json in your results directory)
    results_directory = "results"  # Replace with your actual results directory
    from visualization.visualization_data_extractor import extract_metrics_from_results
    visualization_data = extract_metrics_from_results(results_directory)

    if visualization_data:
        create_duration_boxplot_file_formats(visualization_data, results_directory, noise_level=1000)
        create_duration_boxplot_file_formats(visualization_data, results_directory, noise_level=5000)
    else:
        print("No visualization data found.  Run tests or ensure results are in the correct directory.")
