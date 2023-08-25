import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

"""
Plotting functions for history of genetic algorithm
History is a dictionary containing parallel lists
Index in lists corresponds to generation
Three current lists are beta, gamma, and candidate_populations
"""


def get_n_colors(n, cspace="viridis"):
    cmap = cm.get_cmap(cspace)
    colors = [cmap(i / n) for i in range(n)]
    colors.reverse()
    return colors


def plot_history_of_beta_and_gamma(history, metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=[10 * len(metrics), 4])
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        plot_density_history_of_metric(history, metric, 10, axes[i])

    return fig


def plot_density_history_of_metric(history, metric, n_steps, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    n_generations = len(history[f"{metric}_balance"])
    colors = get_n_colors(n_generations)
    if n_steps > n_generations:
        n_steps = n_generations
    for g in range(0, n_generations, int(n_generations / n_steps)):
        sns.kdeplot(
            -1 * history[f"{metric}_balance"][g],
            ax=ax,
            color=colors[g],
            label=f"Gen {g}",
        )
    sns.kdeplot(
        -1 * history[f"{metric}_balance"][n_generations - 1],
        ax=ax,
        color=colors[g],
        label=f"Gen {n_generations-1}",
    )
    ax.set_xlabel(f"{metric}")
    ax.set_title(f"History of {metric} scores over generations")
    ax.legend()
    ax.grid("on")


def plot_convergence(*histories, labels=None, ax=None, metric="gamma"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[12, 6])

    colors = get_n_colors(len(histories))
    label = None
    for exp, history in enumerate(histories):
        best_candidate = -np.array(histories[exp][f"{metric}_balance"]).max(axis=1)
        if labels:
            label = labels[exp]

        ax.plot(best_candidate, label=label, c=colors[exp])
    ax.set_xlabel("generation")
    ax.set_ylabel(f"best {metric}")
    ax.set_title(f"Best {metric} score over generations")
    ax.grid()
    fig.legend()
    return fig
