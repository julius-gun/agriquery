# python -m visualization.plot_scripts.line_charts

import matplotlib.pyplot as plt
import os
from typing import Dict, List
import numpy as np

ALLOWED_NOISE_LEVELS_WARNING = [1000, 10000, 30000]


def create_line_chart(
    visualization_data: Dict,
    output_dir: str,
    metric: str,
    file_extension: str,
    language: str = None,  # Optional language filter
    dataset: str = None,  # Optional dataset filter
    language_aggregated: bool = False,  # New parameter to control chart type
):
    """
    Creates a line chart showing performance metrics vs. noise level.
    Supports different chart types based on 'language_aggregated' parameter.
    - If language_aggregated=False (default): Creates line chart for specific language/dataset or averages.
    - If language_aggregated=True: Creates line chart with languages differentiated by line style for a given file extension.
    Direct labels are used at the end of each line.

    Args:
        visualization_data (Dict): The data structure containing extracted metrics.
        output_dir (str): The directory to save the plot.
        metric (str): The performance metric to plot (accuracy, f1_score, precision, recall, duration_mean).
        file_extension (str): The file extension to filter data for (e.g., "txt", "csv").
        language (str, optional): Language to filter data for (for non-aggregated chart). If None, averages across all languages.
        dataset (str, optional): Dataset to filter data for (for non-aggregated chart). If None, averages across all datasets.
        language_aggregated (bool): If True, creates language-aggregated chart, otherwise creates filtered/averaged chart.
    """
    if language_aggregated:
        noise_levels_to_plot, plot_data = _prepare_line_chart_data_aggregated_language(
            visualization_data, metric, file_extension
        )
        if plot_data:  # Only plot if there is data
            _plot_line_chart_aggregated_language(
                output_dir,
                metric,
                file_extension,
                plot_data,
                noise_levels_to_plot,
            )
        else:
            print(
                f"No data to plot for aggregated language line chart: metric: {metric}, file extension: {file_extension}"
            )

    else:  # Original filtered/averaged line chart
        noise_levels_to_plot, plot_data = _prepare_line_chart_data_filtered(
            visualization_data, metric, file_extension, language, dataset
        )

        if not plot_data:
            print(
                f"No data to plot for metric: {metric}, file extension: {file_extension}, language: {language}, dataset: {dataset}"
            )
            return

        _plot_line_chart_filtered(
            output_dir,
            metric,
            file_extension,
            language,
            dataset,
            plot_data,
            noise_levels_to_plot,
        )


def _prepare_line_chart_data_filtered(
    visualization_data, metric, file_extension, language, dataset
):
    """Prepares data for the filtered line chart, including filtering and averaging."""
    all_noise_levels = [1000, 2000, 5000, 10000, 20000, 30000, 59000]
    models_data = {}
    common_noise_levels = set(all_noise_levels)

    print(f"\n--- _prepare_line_chart_data_filtered ---") # Function entry marker
    print(f"Metric: {metric}, File Extension: {file_extension}, Language: {language}, Dataset: {dataset}") # Input parameters
    print(f"Initial all_noise_levels: {all_noise_levels}") # Print initial noise levels

    # Determine common noise levels available across all models
    common_noise_levels = set(all_noise_levels)  # start with all possible noise levels
    models_with_data_count = 0  # counter for models with valid data
    common_noise_levels, models_data = _determine_common_noise_levels_and_model_data(
        visualization_data,
        file_extension,
        dataset,
        language,
        metric,
        common_noise_levels,
        models_data,
    )

    print(f"common_noise_levels after _determine_common_noise_levels_and_model_data: {common_noise_levels}") # Print common noise levels after processing

    noise_levels_to_plot = sorted(list(common_noise_levels))

    # Filter plot_data to include only the common noise levels
    if dataset:
        if language and language in models_data:
            plot_data = models_data[language]
        else:
            plot_data = models_data.get("average", {})
    else:
        plot_data = models_data.get("average_dataset", {})

    print(f"noise_levels_to_plot before filtering plot_data: {noise_levels_to_plot}") # Noise levels before plot_data filtering

    # Ensure plot_data only contains common noise levels
    for model_name, metrics in plot_data.items():
        plot_data[model_name] = [
            value
            for nl, value in zip(all_noise_levels, metrics)
            if nl in noise_levels_to_plot
        ]

    print(f"noise_levels_to_plot after filtering plot_data: {noise_levels_to_plot}") # Noise levels after plot_data filtering

    # Filter noise_levels_to_plot to only include levels for which there is data in plot_data
    if plot_data:
        first_model_name = next(
            iter(plot_data)
        )  # Get the first model to check noise levels
        if plot_data[first_model_name]:  # Check if there is data for the first model
            noise_levels_to_plot = noise_levels_to_plot[
                : len(plot_data[first_model_name])
            ]  # Truncate noise_levels_to_plot
    print(f"Final noise_levels_to_plot: {noise_levels_to_plot}") # Print final noise levels to be plotted


    return noise_levels_to_plot, plot_data


def _determine_common_noise_levels_and_model_data(
    visualization_data,
    file_extension,
    dataset,
    language,
    metric,
    common_noise_levels,
    models_data,
):
    """Determines common noise levels and prepares model data, handling dataset and language averaging."""
    models_with_data_count = 0 # Initialize models_with_data_count here
    for model_name, model_results in visualization_data.items():
        if file_extension in model_results:
            if dataset:  # Dataset filter is applied
                common_noise_levels, models_with_data_count = (
                    _process_dataset_filtered_data(
                        model_results,
                        file_extension,
                        language,
                        model_name,
                        dataset,
                        metric,
                        common_noise_levels,
                        models_data,
                        models_with_data_count,
                    )
                )
            else:  # Average across datasets
                common_noise_levels, models_data, models_with_data_count = (
                    _process_dataset_averaged_data(
                        model_results,
                        file_extension,
                        model_name,
                        metric,
                        language,
                        common_noise_levels,
                        models_data,
                        models_with_data_count,
                    )
                )
    return common_noise_levels, models_data


def _prepare_line_chart_data_aggregated_language(
    visualization_data, metric, file_extension
):
    """Prepares data for the line chart, aggregated by language, for a given file extension."""
    all_noise_levels = [1000, 2000, 5000, 10000, 20000, 30000, 59000]
    plot_data = {}
    common_noise_levels = set() # Initialize as empty set for union
    model_names = set() # To store unique model names

    # Determine common noise levels available across all models and languages for the given file extension
    common_noise_levels = set()
    models_with_data_count = 0

    for model_name, model_results in visualization_data.items():
        if file_extension in model_results:
            model_names.add(model_name) # Collect model names
            file_extension_data = model_results[file_extension]
            model_noise_levels_all_languages = set()
            has_language_data = False

            for dataset_name in file_extension_data: # Iterate over datasets
                if isinstance(file_extension_data[dataset_name], dict): # Check if dataset_name is a valid dataset dict
                    for language in file_extension_data[dataset_name]: # Iterate over languages within each dataset
                        language_data = file_extension_data[dataset_name][language]
                        lang_noise_levels = set(language_data.keys())
                        if lang_noise_levels:
                            model_noise_levels_all_languages.update(lang_noise_levels)
                            has_language_data = True

            if has_language_data:
                common_noise_levels.update(model_noise_levels_all_languages) # Use union instead of intersection
                models_with_data_count += 1

    noise_levels_to_plot = sorted(list(common_noise_levels))
    if not model_names:
        return noise_levels_to_plot, plot_data # Return empty plot_data if no models

    for language in ["english", "german", "french"]:  # Fixed languages
        if language not in plot_data:
            plot_data[language] = {} # Initialize language dict
        for model_name in model_names: # Iterate over collected model names
            language_model_metric_values = []
            has_metric_value_for_language = False  # Flag for each language
            if file_extension in visualization_data[model_name]: # Check file extension again to be safe
                file_extension_data = visualization_data[model_name][file_extension]


                for nl in noise_levels_to_plot:  # Use common noise levels
                    metric_value_lang_nl = np.nan  # Default to NaN if no data
                    nl_str = str(nl) # Convert noise level to string for dict access

                    # Iterate through datasets to find metric value for the current language and noise level
                    for dataset_name in file_extension_data:
                        if (
                            dataset_name in file_extension_data
                            and language in file_extension_data[dataset_name]
                            and nl_str in file_extension_data[dataset_name][language]
                        ):
                            metric_value_lang_nl = file_extension_data[dataset_name][
                                language
                            ][nl_str].get(metric, np.nan)
                            if not np.isnan(
                                metric_value_lang_nl
                            ):  # Use first valid metric value found
                                break  # Stop searching datasets if value is found

                    language_model_metric_values.append(metric_value_lang_nl)
                    if not np.isnan(metric_value_lang_nl):
                        has_metric_value_for_language = (
                            True  # Set flag if any valid value found
                        )

                if has_metric_value_for_language:  # Only add to plot data if there is at least one valid metric value for the language
                    if language not in plot_data:
                        plot_data[language] = {}
                    plot_data[language][model_name] = language_model_metric_values

    noise_levels_to_plot = sorted(list(common_noise_levels))

    return noise_levels_to_plot, plot_data


def _plot_line_chart_filtered(
    output_dir,
    metric,
    file_extension,
    language,
    dataset,
    plot_data,
    noise_levels_to_plot,  # Use noise_levels_to_plot here
):
    """Plots the line chart with direct labels and saves it to a file."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(right=0.85)
    model_names = list(plot_data.keys())
    num_models = len(model_names)
    color_cycle = plt.cm.get_cmap("viridis", num_models)  # More distinct colors

    label_offset_x_factor = 100  # Control horizontal jitter for labels
    for i, model_name in enumerate(model_names):
        y_values = plot_data[model_name]
        if (
            not noise_levels_to_plot
            or not y_values
            or len(noise_levels_to_plot) != len(y_values)
        ):  # Check if noise_levels or y_values is empty or lengths mismatch
            print(
                f"Warning: Insufficient or mismatched data points to draw line for model {model_name}, metric {metric}, file extension {file_extension}, language {language}, dataset {dataset}. Skipping line plot."
            )
            continue

        (line,) = ax.plot(
            noise_levels_to_plot,  # Use noise_levels_to_plot for x-axis
            y_values,
            marker="o",
            linestyle="-",
            color=color_cycle(i),
            label=model_name,
        )

        last_x = noise_levels_to_plot[-1]  # Use noise_levels_to_plot here
        last_y = y_values[-1]

        # Check if the last y-value is NaN, if so, find the last valid y-value and its x-position
        if np.isnan(last_y):
            valid_indices = np.where(~np.isnan(y_values))[0]
            if len(valid_indices) > 0:
                last_valid_index = valid_indices[-1]
                last_y = y_values[last_valid_index]
                last_x = noise_levels_to_plot[
                    last_valid_index
                ]  # corrected index # Use noise_levels_to_plot here
            else:
                continue  # Skip labeling if all y_values are NaN

        # Dynamic horizontal alignment based on x position
        ha = "right"
        offset_x = label_offset_x_factor + (
            i * 15
        )  # jitter offset based on model index
        # offset_y = 0 # can add vertical offset if needed

        # Corrected ax.text call - removed xytext and textcoords
        ax.text(
            last_x + offset_x,
            last_y,
            f" {model_name}",
            color=line.get_color(),
            va="center",
            ha=ha,
            clip_on=True,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )  # Clip labels and add background
    _set_chart_labels_and_title_filtered(
        ax,
        metric,
        file_extension,
        language,
        dataset,
        noise_levels_to_plot,  # Use noise_levels_to_plot here
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.95, 1])

    _save_line_chart_filtered(
        output_dir, metric, file_extension, language, dataset, fig
    )


def _process_dataset_filtered_data(
    model_results,
    file_extension,
    language,
    model_name,
    dataset,
    metric,
    common_noise_levels,
    models_data,
    models_with_data_count,
):
    """Processes data when a specific dataset is filtered."""

    if dataset in model_results[file_extension]:
        dataset_data = model_results[file_extension][dataset]
    else:
        _issue_dataset_warning(
            dataset,
            model_name,
            file_extension,
        )
        return (
            common_noise_levels,
            models_with_data_count,
        )  # Return early if dataset is not found

    if language:
        if language in dataset_data:
            model_noise_levels = set(dataset_data[language].keys())
            if model_noise_levels:
                # check if model has data for this language
                common_noise_levels.update(model_noise_levels)  # Use union to include all noise levels
                models_with_data_count += 1
            else:
                _issue_noise_level_warning(
                    model_name,
                    language,
                    dataset,
                    file_extension,
                    "language",
                )
        else:
            _issue_language_warning(language, model_name, dataset, file_extension)

    else:  # Average across languages for the dataset
        model_noise_levels = set()
        has_language_data = False  # flag to check if model has data for any language
        for lang in dataset_data:
            lang_noise_levels = set(dataset_data[lang].keys())
            if lang_noise_levels:  # check if language has data
                model_noise_levels.update(
                    lang_noise_levels
                )  # collect all noise levels from all languages
                has_language_data = True  # set flag if any language has data
        if (
            has_language_data
        ):  # only consider models that have data for at least one language
            common_noise_levels.update(model_noise_levels)  # Use union to include all noise levels
            models_with_data_count += 1
        else:
            _issue_no_language_data_warning(model_name, dataset, file_extension)

    if language:
        models_data = _collect_model_metric_values_language_filtered(
            dataset_data,
            language,
            model_name,
            metric,
            noise_levels_to_plot=sorted(list(common_noise_levels)),
            models_data=models_data,
        )
    else:
        models_data = _collect_model_metric_values_language_averaged(
            dataset_data,
            model_name,
            metric,
            noise_levels_to_plot=sorted(list(common_noise_levels)),
            models_data=models_data,
        )
    return common_noise_levels, models_with_data_count


def _process_dataset_averaged_data(
    model_results,
    file_extension,
    model_name,
    metric,
    language,
    common_noise_levels,
    models_data,
    models_with_data_count,
):
    """Processes data when averaging across datasets."""
    dataset_noise_levels_all_datasets = set()
    has_dataset_data = False  # flag to check if model has data for any dataset
    for dataset_name in model_results[file_extension]:
        dataset_data = model_results[file_extension][dataset_name]
        dataset_noise_levels = set() # Initialize dataset_noise_levels here
        if language: # Language filter is applied
            if language in dataset_data:
                dataset_noise_levels = set(dataset_data[language].keys())
                if dataset_noise_levels:
                    dataset_noise_levels_all_datasets.update(dataset_noise_levels)
                    has_dataset_data = True
                else:
                    _issue_noise_level_warning(
                        model_name,
                        language,
                        dataset_name,
                        file_extension,
                        "language",
                    )
            else:
                _issue_language_warning_dataset_average(
                    language, dataset_name, model_name, file_extension
                )

        else:  # Average across languages
            for lang in dataset_data:
                lang_noise_levels = set(dataset_data[lang].keys())
                if lang_noise_levels:
                    dataset_noise_levels.update(lang_noise_levels) # Now dataset_noise_levels is initialized
                    dataset_noise_levels_all_datasets.update(dataset_noise_levels)
                    has_dataset_data = True
    if has_dataset_data:
        common_noise_levels.update(dataset_noise_levels_all_datasets) # Use union to include all noise levels
        models_with_data_count += 1
    else:
        _issue_no_dataset_data_warning(model_name, file_extension)
    if language:
        models_data = (
            _collect_model_metric_values_language_dataset_averaged_language_filtered(
                model_results,
                file_extension,
                language,
                model_name,
                metric,
                noise_levels_to_plot=sorted(list(common_noise_levels)),
                models_data=models_data,
            )
        )
    else:
        models_data = _collect_model_metric_values_language_dataset_averaged(
            model_results,
            file_extension,
            model_name,
            metric,
            noise_levels_to_plot=sorted(list(common_noise_levels)),
            models_data=models_data,
        )
    return common_noise_levels, models_data, models_with_data_count


def _collect_model_metric_values_language_filtered(
    dataset_data, language, model_name, metric, noise_levels_to_plot, models_data
):
    """Collects metric values for language-filtered data."""
    if language not in models_data:
        models_data[language] = {}
    model_metric_values = []
    for nl in noise_levels_to_plot:
        if language in dataset_data and nl in dataset_data[language]: # Check if language exists in dataset_data
            metric_value = dataset_data[language][nl].get(metric, np.nan)
        else:
            metric_value = np.nan
        model_metric_values.append(metric_value)


    if not all(np.isnan(model_metric_values)):
        models_data[language][model_name] = model_metric_values
    else:
        _issue_no_valid_metric_values_warning(
            model_name,
            language,
            dataset_data,
            file_extension="",
        )
    return models_data


def _collect_model_metric_values_language_averaged(
    dataset_data, model_name, metric, noise_levels_to_plot, models_data
):
    """Collects metric values for language-averaged data."""
    language_metrics = []
    has_valid_language_metric = (
        False  # flag to check if there is any valid language metric
    )
    for lang in dataset_data:
        lang_metric_values = [
            dataset_data[lang].get(nl, {}).get(metric, np.nan)
            if nl in dataset_data[lang]
            and metric
            in dataset_data[lang][nl]  # check if noise level and metric exist
            else np.nan  # explicitly set to NaN if data is missing
            for nl in noise_levels_to_plot
        ]
        if not all(
            np.isnan(lang_metric_values)
        ):  # only consider languages with at least one valid metric value
            language_metrics.append(lang_metric_values)
            has_valid_language_metric = (
                True  # set flag if any valid language metric found
            )
    if (
        has_valid_language_metric
    ):  # only proceed if there is at least one language with valid metric
        average_metrics = np.nanmean(
            language_metrics, axis=0
        )  # Calculate mean across languages
        if "average" not in models_data:
            models_data["average"] = {}
        models_data["average"][model_name] = average_metrics
    else:
        _issue_no_valid_metric_values_warning(
            model_name,
            language="average",
            dataset_data=dataset_data,
            file_extension="",
        )
    return models_data


def _collect_model_metric_values_language_dataset_averaged_language_filtered(
    model_results,
    file_extension,
    language,
    model_name,
    metric,
    noise_levels_to_plot,
    models_data,
):
    """Collects metric values for language-filtered data when averaging across datasets."""
    dataset_metrics_all_datasets = []
    has_valid_dataset_metric = False
    for dataset_name in model_results[file_extension]:
        dataset_data = model_results[file_extension][dataset_name]
        if language:
            if language in dataset_data:
                dataset_metric_values = [
                    dataset_data[language].get(nl, {}).get(metric, np.nan)
                    if nl in dataset_data[language]
                    and metric
                    in dataset_data[language][
                        nl
                    ]  # check if noise level and metric exist
                    else np.nan  # explicitly set to NaN if data is missing
                    for nl in noise_levels_to_plot
                ]
                if not all(np.isnan(dataset_metric_values)):
                    dataset_metrics_all_datasets.append(dataset_metric_values)
                    has_valid_dataset_metric = True
    if has_valid_dataset_metric:
        average_dataset_metrics = np.nanmean(dataset_metrics_all_datasets, axis=0)
        if (
            language not in models_data
        ):  # Use language as key for dataset averaged data when language filter is applied
            models_data[language] = {}
        models_data[language][model_name] = average_dataset_metrics

    else:
        _issue_no_valid_dataset_metric_values_warning(
            model_name,
            file_extension,
            language,
        )
    return models_data


def _collect_model_metric_values_language_dataset_averaged(
    model_results, file_extension, model_name, metric, noise_levels_to_plot, models_data
):
    """Collects metric values for language and dataset averaged data."""
    dataset_metrics_all_datasets = []
    has_valid_dataset_metric = False
    for dataset_name in model_results[file_extension]:
        dataset_data = model_results[file_extension][dataset_name]
        language_metrics = []
        has_valid_language_metric = False
        for lang in dataset_data:
            lang_metric_values = [
                dataset_data[lang].get(nl, {}).get(metric, np.nan)
                if nl in dataset_data[lang]
                and metric
                in dataset_data[lang][nl]  # check if noise level and metric exist
                else np.nan  # explicitly set to NaN if data is missing
                for nl in noise_levels_to_plot
            ]
            if not all(np.isnan(lang_metric_values)):
                language_metrics.append(lang_metric_values)
                has_valid_language_metric = True
        if has_valid_language_metric:
            average_language_metrics = np.nanmean(language_metrics, axis=0)
            dataset_metrics_all_datasets.append(average_language_metrics)
            has_valid_dataset_metric = True

    if has_valid_dataset_metric:
        average_dataset_metrics = np.nanmean(dataset_metrics_all_datasets, axis=0)
        if "average_dataset" not in models_data:
            models_data["average_dataset"] = {}
        models_data["average_dataset"][model_name] = average_dataset_metrics
    else:
        _issue_no_valid_dataset_metric_values_warning(
            model_name,
            file_extension,
            language="average",
        )
    return models_data


def _issue_noise_level_warning(
    model_name, language, dataset, file_extension, level_type
):
    """Issues a warning about missing noise level data, conditional on noise levels."""
    print(
        f"Warning: No data found for model {model_name}, language {language}, dataset {dataset}, file extension {file_extension}. Skipping model in line chart."
    )


def _issue_language_warning(language, model_name, dataset, file_extension):
    """Issues a warning about missing language data."""
    print(
        f"Warning: No language '{language}' found for model '{model_name}', dataset '{dataset}', file extension '{file_extension}'. Skipping language for this model."
    )


def _issue_no_language_data_warning(model_name, dataset, file_extension):
    """Issues a warning about no language data found for a model."""
    print(
        f"Warning: No language data found for model {model_name}, dataset {dataset}, file extension {file_extension}. Skipping model in average line chart."
    )


def _issue_dataset_warning(dataset, model_name, file_extension):
    """Issues a warning about missing dataset, conditional on noise levels."""
    print(
        f"Warning: No dataset {dataset} found for model {model_name}, file extension {file_extension}. Skipping dataset for this model."
    )


def _issue_language_warning_dataset_average(
    language, dataset_name, model_name, file_extension
):
    """Issues a warning about missing language data when averaging across datasets."""
    print(
        f"Warning: No language {language} found in dataset {dataset_name} for model {model_name}, file extension {file_extension}. Skipping language in dataset averaging."
    )


def _issue_no_dataset_data_warning(model_name, file_extension):
    """Issues a warning about no dataset data found for a model."""
    print(
        f"Warning: No dataset data found for model {model_name}, file extension {file_extension}. Skipping model in average line chart across datasets."
    )


def _issue_no_valid_metric_values_warning(
    model_name, language, dataset_data, file_extension
):
    """Issues a warning about no valid metric values found, conditional on noise levels."""
    print(
        f"Warning: No valid metric values found for model {model_name}, language {language}, dataset {dataset_data}, file extension {file_extension} for common noise levels. Skipping model in line chart."
    )


def _issue_no_valid_dataset_metric_values_warning(model_name, file_extension, language):
    """Issues a warning about no valid dataset metric values found, conditional on noise levels."""
    print(
        f"Warning: No valid dataset metric values found for model {model_name}, file extension {file_extension}, language {language} for common noise levels. Skipping model in average line chart across datasets."
    )


def _plot_line_chart_aggregated_language(
    output_dir,
    metric,
    file_extension,
    plot_data,
    noise_levels_to_plot,
):
    """Plots the line chart with direct labels and saves it to a file, differentiating languages by line style."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(right=0.85)  # to make space for labels

    language_styles = {
        "english": {"linestyle": "-", "label": "English"},
        "german": {"linestyle": "--", "label": "German"},
        "french": {"linestyle": "-.", "label": "French"},
    }

    model_names_by_language = {
        lang: list(models.keys()) for lang, models in plot_data.items()
    }
    num_models_total = sum(len(models) for models in plot_data.values())
    color_cycle = plt.cm.get_cmap(
        "viridis", num_models_total
    )  # Color map for all models
    color_index = 0  # Index to iterate through colors

    for language, models in plot_data.items():
        style = language_styles[language]
        for model_name, y_values in models.items():
            if (
                not noise_levels_to_plot
                or not y_values
                or len(noise_levels_to_plot) != len(y_values)
            ):  # Check for empty data
                print(
                    f"Warning: Insufficient data points for model {model_name}, language {language}, metric {metric}, file extension {file_extension}. Skipping line plot."
                )
                continue

            (line,) = ax.plot(
                noise_levels_to_plot,
                y_values,
                marker="o",
                linestyle=style["linestyle"],  # Apply language-specific linestyle
                color=color_cycle(color_index),  # Get color from color cycle
                label=None,  # Remove legend labels
            )
            color_index += 1  # Increment color index for next model

            last_x = noise_levels_to_plot[-1]
            last_y = y_values[-1]

            # Find last valid y value if last value is NaN
            if np.isnan(last_y):
                valid_indices = np.where(~np.isnan(y_values))[0]
                if valid_indices.size > 0:
                    last_valid_index = valid_indices[-1]
                    last_y = y_values[last_valid_index]
                    last_x = noise_levels_to_plot[last_valid_index]
                else:
                    continue  # Skip label if all values are NaN

            # Direct labeling - model name and language
            label_text = f"{model_name} ({language.capitalize()})"
            ax.text(
                last_x + 100,  # x-position for label
                last_y,  # y-position for label
                label_text,
                color=line.get_color(),
                va="center",
                ha="left",
                bbox=dict(
                    facecolor="white", alpha=0.7, edgecolor="none", pad=1
                ),  # Label background
            )

    _set_chart_labels_and_title_aggregated_language(
        ax,
        metric,
        file_extension,
        noise_levels_to_plot,
    )
    # Create custom legend for line styles - outside the loop
    handles = [
        plt.Line2D([], [], linestyle=style["linestyle"], color="black")
        for style in language_styles.values()
    ]
    labels = [style["label"] for style in language_styles.values()]
    ax.legend(
        handles, labels, title="Language", loc="upper right", bbox_to_anchor=(1.0, 1.0)
    )  # Add legend for languages

    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to fit labels

    _save_line_chart_aggregated_language(output_dir, metric, file_extension, fig)


def _set_chart_labels_and_title_filtered(
    ax, metric, file_extension, language, dataset, noise_levels
):
    """Sets labels and title for the filtered line chart."""
    ax.set_xlabel("Noise Level (Tokens)")
    ax.set_ylabel(metric.replace("_", " ").title())
    title_suffix = _get_title_suffix(language, dataset)
    ax.set_title(
        f"{metric.replace('_', ' ').title()} vs. Noise Level for {file_extension.upper()} Files {title_suffix}"
    )
    ax.set_xticks(noise_levels)
    ax.set_xticklabels(
        [str(nl) for nl in noise_levels]
    )  # Ensure noise levels are strings on x-axis


def _plot_line_chart(
    output_dir,
    metric,
    file_extension,
    language,
    dataset,
    plot_data,
    noise_levels_to_plot,  # Use noise_levels_to_plot here
):
    """Plots the line chart with direct labels and saves it to a file."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(right=0.85)
    model_names = list(plot_data.keys())
    num_models = len(model_names)
    color_cycle = plt.cm.get_cmap("viridis", num_models)  # More distinct colors

    label_offset_x_factor = 100  # Control horizontal jitter for labels
    for i, model_name in enumerate(model_names):
        y_values = plot_data[model_name]
        if (
            not noise_levels_to_plot
            or not y_values
            or len(noise_levels_to_plot) != len(y_values)
        ):  # Check if noise_levels or y_values is empty or lengths mismatch
            print(
                f"Warning: Insufficient or mismatched data points to draw line for model {model_name}, metric {metric}, file extension {file_extension}, language {language}, dataset {dataset}. Skipping line plot."
            )
            continue

        (line,) = ax.plot(
            noise_levels_to_plot,  # Use noise_levels_to_plot for x-axis
            y_values,
            marker="o",
            linestyle="-",
            color=color_cycle(i),
            label=model_name,
        )

        last_x = noise_levels_to_plot[-1]  # Use noise_levels_to_plot here
        last_y = y_values[-1]

        # Check if the last y-value is NaN, if so, find the last valid y-value and its x-position
        if np.isnan(last_y):
            valid_indices = np.where(~np.isnan(y_values))[0]
            if len(valid_indices) > 0:
                last_valid_index = valid_indices[-1]
                last_y = y_values[last_valid_index]
                last_x = noise_levels_to_plot[
                    last_valid_index
                ]  # corrected index # Use noise_levels_to_plot here
            else:
                continue  # Skip labeling if all y_values are NaN

        # Dynamic horizontal alignment based on x position
        ha = "right"
        offset_x = label_offset_x_factor + (
            i * 15
        )  # jitter offset based on model index
        # offset_y = 0 # can add vertical offset if needed

        # Corrected ax.text call - removed xytext and textcoords
        ax.text(
            last_x + offset_x,
            last_y,
            f" {model_name}",
            color=line.get_color(),
            va="center",
            ha=ha,
            clip_on=True,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )  # Clip labels and add background

    _set_chart_labels_and_title(
        ax,
        metric,
        file_extension,
        language,
        dataset,
        noise_levels_to_plot,  # Use noise_levels_to_plot here
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.95, 1])

    _save_line_chart_filtered(
        output_dir, metric, file_extension, language, dataset, fig
    )


def _set_chart_labels_and_title(
    ax, metric, file_extension, language, dataset, noise_levels
):
    """Sets labels and title for the line chart."""
    ax.set_xlabel("Noise Level (Tokens)")
    ax.set_ylabel(metric.replace("_", " ").title())
    title_suffix = _get_title_suffix(language, dataset)
    ax.set_title(
        f"{metric.replace('_', ' ').title()} vs. Noise Level for {file_extension.upper()} Files {title_suffix}"
    )
    ax.set_xticks(noise_levels)
    ax.set_xticklabels(
        [str(nl) for nl in noise_levels]
    )  # Ensure noise levels are strings on x-axis


def _set_chart_labels_and_title_aggregated_language(
    ax, metric, file_extension, noise_levels
):
    """Sets labels and title for the aggregated language line chart."""
    ax.set_xlabel("Noise Level (Tokens)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"{metric.replace('_', ' ').title()} vs. Noise Level for {file_extension.upper()} Files (Language Aggregated)"
    )
    ax.set_xticks(noise_levels)
    ax.set_xticklabels(
        [str(nl) for nl in noise_levels]
    )  # Ensure noise levels are strings on x-axis


def _get_title_suffix(language, dataset):
    """Constructs the title suffix based on language and dataset."""
    if language and dataset:
        return f"({language.capitalize()}, Dataset: {dataset})"
    elif not language and not dataset:
        return "(Average across Languages and Datasets)"
    elif not language and dataset:
        return f"(Average across Languages, Dataset: {dataset})"
    elif language and not dataset:
        return f"({language.capitalize()}, Average across Datasets)"
    return ""


def _save_line_chart_filtered(
    output_dir, metric, file_extension, language, dataset, fig
):
    """Saves the filtered line chart to a file in 'non_language_aggregated' subfolder."""
    plot_folder = os.path.join(
        output_dir, "visualization", "plots", "line_chart", "non_language_aggregated"
    )  # Updated folder path
    os.makedirs(plot_folder, exist_ok=True)
    language_suffix = f"_{language}" if language else "_avg_lang"
    dataset_suffix = f"_{dataset}" if dataset else "_avg_dataset"
    filepath = os.path.join(
        plot_folder,
        f"{metric}_vs_noise_{file_extension}{language_suffix}{dataset_suffix}.png",
    )
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Line chart saved to {filepath}")
    plt.close(fig)


def _save_line_chart_aggregated_language(output_dir, metric, file_extension, fig):
    """Saves the aggregated language line chart to a file in 'language_aggregated' subfolder."""
    plot_folder = os.path.join(
        output_dir, "visualization", "plots", "line_chart", "language_aggregated"
    )  # Updated folder path
    os.makedirs(plot_folder, exist_ok=True)
    filepath = os.path.join(
        plot_folder,
        f"{metric}_vs_noise_{file_extension}_languages.png",  # General name for language differentiated charts
    )
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Line chart saved to {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    # Example Usage (assuming visualization_data is loaded or available)
    results_directory = "results"  # Replace with your actual results directory
    from visualization.visualization_data_extractor import extract_metrics_from_results

    visualization_data = extract_metrics_from_results(results_directory)

    if visualization_data:
        metrics_to_plot = [
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "duration_mean",
        ]
        file_extensions_to_plot = ["txt", "csv", "html"]  # Example file extensions
        languages_to_plot = [
            "english",
            "french",
            "german",
            None,
        ]  # None for average across languages
        datasets_to_plot = [
            "question_answers_pairs.json",
            "question_answers_tables.json",
            "question_answers_unanswerable.json",
            None,
        ]  # Example datasets, None for average across datasets

        for file_extension in file_extensions_to_plot:
            for metric in metrics_to_plot:
                for language in languages_to_plot:
                    for dataset in datasets_to_plot:
                        create_line_chart(
                            visualization_data,
                            results_directory,
                            metric,
                            file_extension,
                            language,
                            dataset,
                            language_aggregated=False,  # Explicitly set to False for old behavior
                        )
                # Generate new language aggregated line charts
                create_line_chart(
                    visualization_data,
                    results_directory,
                    metric,
                    file_extension,
                    language_aggregated=True,  # Set to True for new language aggregated chart
                )

    else:
        print(
            "No visualization data found. Please ensure results are processed and visualization_data is available."
        )
