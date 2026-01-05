"""
plots.py

Plot helpers for summarizing model results in a compact, readable way.
Works directly with the results dicts produced by models.py.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_heatmap_from_results(
    svm_results,
    rf_results,
    metric="f1",
    feature_sets=None,
    title=None,
    figsize=(6, 3.5),
):
    """
    Plot a heatmap comparing SVM and RF across feature sets for a given metric.

    Parameters
    ----------
    svm_results : dict
        Dict produced by models.py for SVM:
        svm_results[scenario][metric_key]
    rf_results : dict
        Dict produced by models.py for RF:
        rf_results[scenario][metric_key]
    metric : str
        One of: 'accuracy', 'balanced_accuracy', 'f1', 'roc_auc'.
    feature_sets : list or None
        Feature-set order to show on the y-axis. If None, uses svm_results keys.
    title : str or None
        Custom title. If None, a default title is used.
    figsize : tuple
        Figure size passed to matplotlib.

    Raises
    ------
    KeyError
        If a scenario or metric key is missing from results.
    """
    if feature_sets is None:
        feature_sets = list(svm_results.keys())

    models = ["SVM", "RF"]
    mat = np.zeros((len(feature_sets), 2), dtype=float)

    for i, fs in enumerate(feature_sets):
        mat[i, 0] = svm_results[fs][metric]
        mat[i, 1] = rf_results[fs][metric]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(models)
    ax.set_yticks(np.arange(len(feature_sets)))
    ax.set_yticklabels(feature_sets)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "{:.3f}".format(mat[i, j]), ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    ax.set_xlabel("Model")
    ax.set_ylabel("Feature set")
    ax.set_title(title or "Test-set {}".format(metric))

    plt.tight_layout()
    plt.show()


def plot_all_metric_heatmaps(
    svm_results,
    rf_results,
    feature_sets=None,
    metrics=None,
):
    """
    Plot a sequence of heatmaps (one per metric) to avoid cherry-picking.

    Parameters
    ----------
    svm_results : dict
        SVM results dict from models.py.
    rf_results : dict
        RF results dict from models.py.
    feature_sets : list or None
        Feature-set order. If None, uses svm_results keys.
    metrics : list or None
        List of metric keys to plot. If None, plots the common four metrics.
    """
    if metrics is None:
        metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]

    for metric in metrics:
        plot_metric_heatmap_from_results(
            svm_results=svm_results,
            rf_results=rf_results,
            metric=metric,
            feature_sets=feature_sets,
            title="Test-set {}".format(metric),
        )


def plot_confusion_matrices_strip(
    results,
    scenarios,
    model_name,
    cm_key="confusion_matrix",
    figsize=None,
):
    """
    Plot confusion matrices side-by-side for selected scenarios.

    Parameters
    ----------
    results : dict
        Results dict from models.py (svm_results OR rf_results).
    scenarios : list
        Scenarios/feature-sets to plot in the given order.
    model_name : str
        Used in subplot titles (e.g., 'SVM' or 'RF').
    cm_key : str
        Key used for confusion matrix in results (default: 'confusion_matrix').
    figsize : tuple or None
        If None, chooses a reasonable width based on number of scenarios.

    Raises
    ------
    KeyError
        If a scenario is missing or cm_key is missing.
    """
    n = len(scenarios)
    if figsize is None:
        figsize = (3 * n, 3)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        cm = results[sc][cm_key]
        im = ax.imshow(cm)

        ax.set_title("{}\n{}".format(model_name, sc))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()
