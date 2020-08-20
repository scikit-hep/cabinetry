import logging
import os
import pathlib
from typing import Any, Dict, List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


log = logging.getLogger(__name__)


def _total_yield_uncertainty(stdev_list: List[np.ndarray]) -> np.ndarray:
    """calculate the absolute statistical uncertainty of a stack of MC
    via sum in quadrature

    Args:
        stdev_list (List[np.ndarray]): list of absolute stat. uncertainty per sample

    Returns:
        np.array: absolute stat. uncertainty of stack of samples
    """
    tot_unc = np.sqrt(np.sum(np.power(stdev_list, 2), axis=0))
    return tot_unc


def data_MC(
    histogram_dict_list: List[Dict[str, Any]], figure_path: pathlib.Path
) -> None:
    """draw a data/MC histogram

    Args:
        histogram_dict_list (List[Dict[str, Any]]): list of samples (with info stored in one dict per sample)
        figure_path (pathlib.Path): path where figure should be saved
    """
    mc_histograms_yields = []
    mc_histograms_stdev = []
    mc_labels = []
    for h in histogram_dict_list:
        if h["isData"]:
            data_histogram_yields = h["hist"]["yields"]
            data_histogram_stdev = h["hist"]["stdev"]
            data_label = h["label"]
        else:
            mc_histograms_yields.append(h["hist"]["yields"])
            mc_histograms_stdev.append(h["hist"]["stdev"])
            mc_labels.append(h["label"])

    # get the highest single bin from the sum of MC
    y_max = np.max(
        np.sum(
            [h["hist"]["yields"] for h in histogram_dict_list if not h["isData"]],
            axis=0,
        )
    )

    # if data is higher in any bin, the maximum y axis range should take that into account
    y_max = max(
        y_max, np.max([h["hist"]["yields"] for h in histogram_dict_list if h["isData"]])
    )

    mpl.style.use("seaborn-colorblind")

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for item in (
        [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label]
        + ax1.get_yticklabels()
        + ax2.get_xticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize("large")

    # plot MC stacked together
    total_yield = np.zeros_like(mc_histograms_yields[0])
    bins = histogram_dict_list[0]["hist"]["bins"]
    bin_right_edges = bins[1:]
    bin_left_edges = bins[:-1]
    bin_width = bin_right_edges - bin_left_edges
    bin_centers = 0.5 * (bin_left_edges + bin_right_edges)
    for i_sample, mc_sample_yield in enumerate(mc_histograms_yields):
        ax1.bar(
            bin_centers,
            mc_sample_yield,
            width=bin_width,
            bottom=total_yield,
            label=mc_labels[i_sample],
        )
        total_yield += mc_sample_yield

    # add total MC uncertainty
    mc_stack_unc = _total_yield_uncertainty(mc_histograms_stdev)
    ax1.bar(
        bin_centers,
        2 * mc_stack_unc,
        width=bin_width,
        bottom=total_yield - mc_stack_unc,
        label="Stat. uncertainty",
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # plot data
    ax1.errorbar(
        bin_centers,
        data_histogram_yields,
        yerr=data_histogram_stdev,
        fmt="o",
        color="k",
        label=data_label,
    )

    # ratio plot
    ax2.plot(
        [bin_left_edges[0], bin_right_edges[-1]],
        [1, 1],
        "--",
        color="black",
        linewidth=1,
    )  # reference line along y=1

    # add uncertainty band around y=1
    rel_mc_unc = mc_stack_unc / total_yield
    ax2.bar(
        bin_centers,
        2 * rel_mc_unc,
        width=bin_width,
        bottom=1 - rel_mc_unc,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # data in ratio plot
    data_model_ratio = data_histogram_yields / total_yield
    data_model_ratio_unc = data_histogram_stdev / total_yield
    ax2.errorbar(
        bin_centers, data_model_ratio, yerr=data_model_ratio_unc, fmt="o", color="k",
    )

    ax1.legend(frameon=False, fontsize="large")
    ax1.set_xlim(bin_left_edges[0], bin_right_edges[-1])
    ax1.set_ylim([0, y_max * 1.5])  # 50% headroom
    ax1.set_ylabel("events")
    ax1.set_xticklabels([])
    ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
    ax1.tick_params(direction="in", top=True, right=True)

    ax2.set_xlim(bin_left_edges[0], bin_right_edges[-1])
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel(histogram_dict_list[0]["variable"])
    ax2.set_ylabel("data / model")
    ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()

    if not os.path.exists(figure_path.parent):
        os.mkdir(figure_path.parent)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)


def correlation_matrix(
    corr_mat: np.ndarray,
    labels: Union[List[str], np.ndarray],
    figure_path: pathlib.Path,
) -> None:
    """draw a correlation matrix

    Args:
        corr_mat (np.ndarray): the correlation matrix to plot
        labels (Union[List[str], np.ndarray]): names of parameters in the correlation matrix
        figure_path (pathlib.Path): path where figure should be saved
    """
    # rounding for test in CI to match reference
    fig, ax = plt.subplots(
        figsize=(round(5 + len(labels) / 1.6, 1), round(3 + len(labels) / 1.6, 1)),
        dpi=100,
    )
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

    fig.colorbar(im, ax=ax)
    ax.set_aspect("auto")  # to get colorbar aligned with matrix
    fig.tight_layout()

    # add correlation as text
    for (j, i), corr in np.ndenumerate(corr_mat):
        text_color = "white" if abs(corr_mat[j, i]) > 0.75 else "black"
        if abs(corr) > 0.005:
            ax.text(i, j, f"{corr:.2f}", ha="center", va="center", color=text_color)

    if not os.path.exists(figure_path.parent):
        os.mkdir(figure_path.parent)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)


def pulls(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: Union[List[str], np.ndarray],
    figure_path: pathlib.Path,
) -> None:
    """draw a pull plot

    Args:
        bestfit (np.ndarray): [description]
        uncertainty (np.ndarray): parameter uncertainties
        labels (Union[List[str], np.ndarray]): parameter names
        figure_path (pathlib.Path): path where figure should be saved
    """
    num_pars = len(bestfit)
    y_positions = np.arange(num_pars)[::-1]
    fig, ax = plt.subplots(figsize=(6, 1 + num_pars / 4), dpi=100)
    ax.errorbar(bestfit, y_positions, xerr=uncertainty, fmt="o", color="black")

    ax.fill_between([-2, 2], -0.5, len(bestfit) - 0.5, color="yellow")
    ax.fill_between([-1, 1], -0.5, len(bestfit) - 0.5, color="limegreen")
    ax.vlines(0, -0.5, len(bestfit) - 0.5, linestyles="dotted", color="black")

    ax.set_xlim([-3, 3])
    ax.set_xlabel(r"$\left(\hat{\theta} - \theta_0\right) / \Delta \theta$")
    ax.set_ylim([-0.5, num_pars - 0.5])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels[::-1])
    fig.tight_layout()

    if not os.path.exists(figure_path.parent):
        os.mkdir(figure_path.parent)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)


def ranking(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: Union[List[str], np.ndarray],
    impact_prefit_up: np.ndarray,
    impact_prefit_down: np.ndarray,
    impact_postfit_up: np.ndarray,
    impact_postfit_down: np.ndarray,
    figure_path: pathlib.Path,
) -> None:
    """Draws a ranking plot.

    Args:
        bestfit (np.ndarray): best-fit parameter results
        uncertainty (np.ndarray): parameter uncertainties
        labels (Union[List[str], np.ndarray]): parameter labels
        impact_prefit_up (np.ndarray): pre-fit impact in "up" direction per parameter
        impact_prefit_down (np.ndarray): pre-fit impact in "down" direction per parameter
        impact_postfit_up (np.ndarray): post-fit impact in "up" direction per parameter
        impact_postfit_down (np.ndarray): post-fit impact in "down" direction per parameter
        figure_path (pathlib.Path): path where figure should be saved
    """
    num_pars = len(bestfit)

    mpl.style.use("seaborn-colorblind")
    fig, ax_pulls = plt.subplots(figsize=(8, 3 + num_pars * 0.4), dpi=100)
    ax_impact = ax_pulls.twiny()  # second x-axis with shared y-axis, used for pulls

    # since pull axis is below impact axis, flip them so pulls show up on top
    ax_pulls.set_zorder(1)  # pulls axis on top
    ax_pulls.patch.set_visible(False)  # pulls axis does not hide impact axis

    # increase font sizes
    for item in (
        [ax_pulls.yaxis.label, ax_pulls.xaxis.label, ax_impact.xaxis.label]
        + ax_pulls.get_yticklabels()
        + ax_pulls.get_xticklabels()
        + ax_impact.get_xticklabels()
    ):
        item.set_fontsize("large")

    y_pos = np.arange(num_pars)[::-1]

    # lines through pulls of -1, 0, 1 for orientation
    # line does not go all the way up to the top x-axis, since it
    # belongs to the bottom x-axis
    ax_pulls.vlines(
        -1, -1, num_pars - 0.5, linestyles="dashed", color="black", linewidth=0.75
    )
    ax_pulls.vlines(
        0, -1, num_pars - 0.5, linestyles="dashed", color="black", linewidth=0.75
    )
    ax_pulls.vlines(
        1, -1, num_pars - 0.5, linestyles="dashed", color="black", linewidth=0.75
    )

    ax_impact.barh(
        y_pos,
        impact_prefit_up,
        label="pre-fit impact up",
        fill=False,
        linewidth=1,
        edgecolor="C0",
    )  # pre-fit up
    ax_impact.barh(
        y_pos,
        impact_prefit_down,
        label="pre-fit impact down",
        fill=False,
        linewidth=1,
        edgecolor="C5",
    )  # pre-fit down
    ax_impact.barh(
        y_pos, impact_postfit_up, label="post-fit impact up", color="C0",
    )  # post-fit up
    ax_impact.barh(
        y_pos, impact_postfit_down, label="post-fit impact down", color="C5",
    )  # post-fit down
    ax_pulls.errorbar(
        bestfit, y_pos, xerr=uncertainty, label="pulls", fmt="o", color="k"
    )  # nuisance parameter pulls

    ax_pulls.set_xlabel(r"$\left(\hat{\theta} - \theta_0\right) / \Delta \theta$")
    ax_impact.set_xlabel(r"$\Delta \mu$")
    ax_pulls.set_xlim([-2, 2])
    ax_impact.set_xlim([-5, 5])
    ax_pulls.set_ylim([-1, num_pars])

    # impact axis limits: need largest pre-fit impact
    impact_max = np.max(np.abs(impact_prefit_up, impact_prefit_down))
    ax_impact.set_xlim([-impact_max * 1.1, impact_max * 1.1])

    # minor ticks
    for axis in [ax_pulls.xaxis, ax_impact.xaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax_pulls.set_yticks(y_pos)
    ax_pulls.set_yticklabels(labels[::-1])

    ax_pulls.tick_params(direction="in", which="both")
    ax_impact.tick_params(direction="in", which="both")

    fig.legend(frameon=False, loc="upper left", ncol=2, fontsize="large")
    leg_space = 0.05 + 0.15 / num_pars
    fig.tight_layout(rect=[0, 0, 1.0, 1 - leg_space])  # make space for legend on top

    if not os.path.exists(figure_path.parent):
        os.mkdir(figure_path.parent)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
