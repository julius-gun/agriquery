# visualization/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def _generate_plot_filename(
    output_dir,
    metric_name,
    model_name,
    file_extension,
    context_type,
    language,
    dataset_type=None,
):
    """Generates a consistent filename for plots."""
    base_filename = (
        f"{language}_{metric_name}_{model_name}_{file_extension}_{context_type}"
    )
    if dataset_type:
        base_filename += f"_{dataset_type}"
    plot_filename = f"{base_filename}.png"
    return os.path.join(output_dir, plot_filename)


def create_boxplots(
    results,
    output_dir,
    metric_name,
    y_axis_label,
    filename_prefix,
    hue_parameter=None,
    dataset_type=None,
):
    """
    Creates box plots for a given metric vs. context size, grouped by hue_parameter.

    Args:
        results (DataFrame): Pandas DataFrame of results.  *MUST* be a DataFrame now.
        output_dir (str): Directory to save the plot.
        metric_name (str): Name of the metric being plotted (for filename and title).
        y_axis_label (str): Label for the y-axis.
        filename_prefix (str): Prefix for the filename.
        hue_parameter (str, optional): Parameter to group boxplots by (e.g., 'model_name', 'language', 'file_extension', 'dataset'). Defaults to None.
        dataset_type (str, optional): Dataset type for filename, if applicable. Defaults to None.
    """
    if results.empty:
        print(f"No results to plot for {metric_name}.")
        return

    # --- No need to create DataFrame here, it's passed in ---

    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        x="noise_level",
        y=metric_name,
        data=results,
        hue=hue_parameter,
        showfliers=False,
    )

    if hue_parameter:
        plt.title(f"{y_axis_label} vs. Context Size, grouped by {hue_parameter}")
    else:
        plt.title(f"{y_axis_label} vs. Context Size")

    plt.xlabel("Context Size (Noise Level)")
    plt.ylabel(y_axis_label)
    if hue_parameter:  # prevent error if hue_parameter is None
        plt.legend(title=hue_parameter)

    # Adjust y-axis for metrics like accuracy, precision, recall, f1_score
    if (
        metric_name in ["accuracy", "precision", "recall", "f1_score"]
        and metric_name in results.columns
    ):
        if not results[metric_name].isnull().all():
            plt.ylim(0, 1.05)

    # --- Get values for filename from the group's data (assuming consistent within group) ---
    if not results.empty:
        language = results['language'].iloc[0]
        model_name = results['model_name'].iloc[0]
        file_extension = results['file_extension'].iloc[0]
        context_type = results['context_type'].iloc[0]
    else: # Handle empty DataFrame case
        language, model_name, file_extension, context_type = "unknown", "unknown", "unknown", "unknown"

    plot_filename = _generate_plot_filename(
        output_dir,
        filename_prefix,
        model_name,
        file_extension,
        context_type,
        language,
        dataset_type,
    )

    plt.savefig(plot_filename)
    plt.close()
    print(f"Box plot saved to {plot_filename}")


def create_duration_boxplots(
    results, output_dir, hue_parameter=None, dataset_type=None
):
    """Creates box plots specifically for duration."""
    create_boxplots(
        results=results,
        output_dir=output_dir,
        metric_name="duration",
        y_axis_label="Duration (seconds)",
        filename_prefix="duration_boxplot",
        hue_parameter=hue_parameter,
        dataset_type=dataset_type,
    )


def create_accuracy_boxplots(
    results, output_dir, hue_parameter=None, dataset_type=None
):
    """Creates box plots for accuracy."""
    create_boxplots(
        results=results,
        output_dir=output_dir,
        metric_name="accuracy",
        y_axis_label="Accuracy",
        filename_prefix="accuracy_boxplot",
        hue_parameter=hue_parameter,
        dataset_type=dataset_type,
    )


def create_precision_boxplots(
    results, output_dir, hue_parameter=None, dataset_type=None
):
    """Creates box plots for precision."""
    create_boxplots(
        results=results,
        output_dir=output_dir,
        metric_name="precision",
        y_axis_label="Precision",
        filename_prefix="precision_boxplot",
        hue_parameter=hue_parameter,
        dataset_type=dataset_type,
    )


def create_recall_boxplots(results, output_dir, hue_parameter=None, dataset_type=None):
    """Creates box plots for recall."""
    create_boxplots(
        results=results,
        output_dir=output_dir,
        metric_name="recall",
        y_axis_label="Recall",
        filename_prefix="recall_boxplot",
        hue_parameter=hue_parameter,
        dataset_type=dataset_type,
    )


def create_f1_score_boxplots(
    results, output_dir, hue_parameter=None, dataset_type=None
):
    """Creates box plots for F1 score."""
    create_boxplots(
        results=results,
        output_dir=output_dir,
        metric_name="f1_score",
        y_axis_label="F1 Score",
        filename_prefix="f1_score_boxplot",
        hue_parameter=hue_parameter,
        dataset_type=dataset_type,
    )
