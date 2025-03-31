# visualization/plot_scripts/main_visualization.py
# python -m visualization.plot_scripts.main_visualization
import os
from visualization.visualization_data_extractor import extract_metrics_from_results
from visualization.plot_scripts.bar_charts import (
    create_metric_comparison_bar_chart,
    create_metric_comparison_bar_chart_file_formats,
    create_metric_comparison_bar_chart_datasets,
)
from visualization.plot_scripts.line_charts import create_line_chart
from visualization.plot_scripts.heatmaps import (
    create_heatmap_metric_per_files,
    create_heatmap_llm_vs_file_format,
    create_heatmap_llm_vs_language,
)
from visualization.plot_scripts.box_plots import create_duration_boxplot_file_formats


def main_visualization(results_dir: str):
    """
    Main function to generate visualizations.
    """
    visualization_data = extract_metrics_from_results(results_dir)

    if not visualization_data:
        print(
            "No visualization data found. Run tests or ensure results are in the correct directory."
        )
        return
    # --- Configuration for Visualizations ---

    metrics_bar_charts = [
        "f1_score",
        "accuracy",
        "precision",
        "recall",
    ]  # metrics to iterate through for plots
    file_extensions_bar_charts = ["txt"]  # file extensions for bar charts
    file_extensions_heatmap = [
        "txt",
        "csv",
        "html",
        "json",
        "xml",
        "yaml",
    ]  # file extensions for heatmap
    noise_levels_bar_charts = [1000, 10000, 30000]  # noise levels for bar charts
    noise_levels_box_plots = [1000, 10000, 300000]  # noise levels for box plots
    file_extensions_line_charts = [
        "txt",
        "csv",
        "html",
        "json",
        "xml",
        "yaml",
    ]  # file extensions for line charts
    languages_line_charts = ["english", "french", "german"]  # languages for line charts
    datasets_line_charts = [
        "question_answers_pairs.json",
        "question_answers_tables.json",
        "question_answers_unanswerable.json",
    ]  # Datasets for line charts
    noise_levels_heatmaps = [1000, 10000, 30000]  # Noise levels for heatmaps
    metrics_line_charts = [
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "duration_mean",
    ]  # Metrics for line charts - Reused from top

    # --- Bar Charts ---
    print("Generating Bar Charts...")
    # Metric Comparison Bar Charts (across models, for each metric, noise level, and file extension)
    for metric in metrics_bar_charts:
        for noise_level in noise_levels_bar_charts:
            for file_extension in file_extensions_bar_charts:
                create_metric_comparison_bar_chart(
                    visualization_data, results_dir, metric, noise_level, file_extension
                )

    # File Format Comparison Bar Charts (across file formats, for each metric and noise level)
    for metric in metrics_bar_charts:
        for noise_level in noise_levels_bar_charts:  # Using noise_levels_bar_charts for file format comparison as well for consistency
            create_metric_comparison_bar_chart_file_formats(
                visualization_data, results_dir, metric, noise_level
            )

    # Dataset Comparison Bar Charts (across datasets, for each metric, noise level and file extension)
    for metric in metrics_bar_charts:
        for noise_level in noise_levels_bar_charts:
            for file_extension in file_extensions_bar_charts:
                create_metric_comparison_bar_chart_datasets(
                    visualization_data, results_dir, metric, noise_level, file_extension
                )


    # --- Line Charts ---
    print("Generating Line Charts...")

    # Metric vs. Noise Level Line Charts (for each metric, file extension, language and dataset)

    for file_extension in file_extensions_line_charts:
        for metric in metrics_line_charts:
            for language in languages_line_charts:
                for dataset in datasets_line_charts:
                    create_line_chart(
                        visualization_data,
                        results_dir,
                        metric,
                        file_extension,
                        language,
                        dataset,
                        language_aggregated=False # Explicitly set to False for old behavior
                    )
                create_line_chart(
                    visualization_data,
                    results_dir,
                    metric,
                    file_extension,
                    language,
                    None, # dataset = None means average across datasets
                    language_aggregated=False # Explicitly set to False for non-language aggregated plots
                )
            create_line_chart( # Line charts averaged across languages and datasets
                visualization_data,
                results_dir,
                metric,
                file_extension,
                None, # language = None means average across languages
                None, # dataset = None means average across datasets
                language_aggregated=False # Explicitly set to False for non-language aggregated plots
            )

    # Language Aggregated Line Charts (Line charts with languages differentiated by line style)
    for file_extension in file_extensions_line_charts:
        for metric in metrics_line_charts:
            create_line_chart(
                visualization_data,
                results_dir,
                metric,
                file_extension,
                language_aggregated=True,  # Set to True for new language aggregated chart
            )
    # Original Line Charts (filtered/averaged) - Keep these after aggregated charts for better folder structure in output
    for file_extension in file_extensions_line_charts:
        for metric in metrics_line_charts:
            for language in ["english", "french", "german", None]: # Include None for average across languages
                for dataset in ["question_answers_pairs.json", "question_answers_tables.json", "question_answers_unanswerable.json", None]: # Include None for average across datasets
                    if language is None and dataset is not None: # Skip average language and specific dataset combination
                        continue
                    if language is not None and dataset is None: # Skip specific language and average dataset combination
                        continue
                    create_line_chart(
                        visualization_data,
                        results_dir,
                        metric,
                        file_extension,
                        language=language,
                        dataset=dataset,
                        language_aggregated=False,  # Explicitly set to False for old behavior
                    )


    # --- Heatmaps ---
    print("Generating Heatmaps...")
    
    # Heatmaps of Metric per File Extension and Model (for each metric)
    for metric in metrics_bar_charts:  # Reusing metrics_bar_charts here, could define a separate list if needed
        create_heatmap_metric_per_files(visualization_data, results_dir, metric)

    # Heatmaps of LLM vs File Format and LLM vs Language for F1-Score and specified noise levels
    for noise_level in noise_levels_heatmaps:
        create_heatmap_llm_vs_file_format(
            visualization_data, results_dir, "f1_score", [noise_level]  # Pass noise_level as a list
        )
        create_heatmap_llm_vs_language(
            visualization_data, results_dir, "f1_score", [noise_level]  # Pass noise_level as a list
        )

    # --- Box Plots ---
    print("Generating Box Plots...")
    # Box Plots of Duration across File Formats (for different noise levels)
    for noise_level in noise_levels_box_plots:
        create_duration_boxplot_file_formats(
            visualization_data, results_dir, noise_level
        )

    # --- Future Dataset-Specific Visualizations ---
    # To create visualizations that analyze performance across different datasets
    # (question_answers_pairs.json, question_answers_tables.json, question_answers_unanswerable.json),
    # you can modify the visualization functions to accept and utilize dataset information.
    #
    # Example:
    # def create_metric_comparison_bar_chart_datasets(visualization_data, output_dir, metric, dataset_filter):
    #     # Filter visualization_data to include only results from the specified dataset_filter
    #     # ... plotting logic ...
    #
    # Then, in main_visualization, you would call this function for each dataset:
    # datasets_bar_charts = ["question_answers_pairs.json", "question_answers_tables.json", "question_answers_unanswerable.json"]
    # for metric in metrics_to_plot:
    #     for noise_level in noise_levels_bar_charts:
    #         for file_extension in file_extensions_to_plot_bar:
    #             for dataset in datasets_bar_charts:
    #                 # Modify create_metric_comparison_bar_chart to accept dataset filter if needed
    #                 # create_metric_comparison_bar_chart_datasets(visualization_data, results_dir, metric, noise_level, file_extension, dataset)
    #                 pass # Placeholder for dataset-specific bar chart function

    print("Visualization generation completed.")

if __name__ == "__main__":
    results_directory = "results"  # Replace with the path to your results directory
    main_visualization(results_directory)
