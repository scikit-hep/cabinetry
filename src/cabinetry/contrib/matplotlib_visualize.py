import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


log = logging.getLogger(__name__)


def data_MC(
    histogram_dict_list: List[Dict[str, Any]],
    total_model_unc: np.ndarray,
    bin_edges: np.ndarray,
    figure_path: pathlib.Path,
    log_scale: Optional[bool] = None,
) -> None:
    """Draws a data/MC histogram with uncertainty bands and ratio panel.

    Args:
        histogram_dict_list (List[Dict[str, Any]]): list of samples (with info stored in
            one dict per sample)
        total_model_unc (np.ndarray): total model uncertainty, if specified this is used
            instead of calculating it via sum in quadrature, defaults to None
        bin_edges (np.ndarray): bin edges of histogram
        figure_path (pathlib.Path): path where figure should be saved
        log_scale (Optional[bool], optional): whether to use a logarithmic vertical
            axis, defaults to None (automatically determine whether to use linear or log
            scale)
    """
    mc_histograms_yields = []
    mc_labels = []
    for h in histogram_dict_list:
        if h["isData"]:
            data_histogram_yields = h["yields"]
            data_histogram_stdev = np.sqrt(data_histogram_yields)
            data_label = h["label"]
        else:
            mc_histograms_yields.append(h["yields"])
            mc_labels.append(h["label"])

    mpl.style.use("seaborn-colorblind")

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # increase font sizes
    for item in (
        [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label]
        + ax1.get_yticklabels()
        + ax2.get_xticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks on all axes
    for axis in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    # plot MC stacked together
    total_yield = np.zeros_like(mc_histograms_yields[0])
    bin_right_edges = bin_edges[1:]
    bin_left_edges = bin_edges[:-1]
    bin_width = bin_right_edges - bin_left_edges
    bin_centers = 0.5 * (bin_left_edges + bin_right_edges)
    mc_containers = []
    for mc_sample_yield in mc_histograms_yields:
        mc_container = ax1.bar(
            bin_centers,
            mc_sample_yield,
            width=bin_width,
            bottom=total_yield,
        )
        mc_containers.append(mc_container)

        # add a black line on top of each sample
        line_x = [y for y in bin_edges for _ in range(2)][1:-1]
        line_y = [y for y in (mc_sample_yield + total_yield) for _ in range(2)]
        ax1.plot(line_x, line_y, "-", color="black", linewidth=0.5)

        total_yield += mc_sample_yield

    # add total MC uncertainty
    mc_unc_container = ax1.bar(
        bin_centers,
        2 * total_model_unc,
        width=bin_width,
        bottom=total_yield - total_model_unc,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # plot data
    data_container = ax1.errorbar(
        bin_centers,
        data_histogram_yields,
        yerr=data_histogram_stdev,
        fmt="o",
        color="k",
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
    rel_mc_unc = total_model_unc / total_yield
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
        bin_centers, data_model_ratio, yerr=data_model_ratio_unc, fmt="o", color="k"
    )

    # get the highest single bin yield, from the sum of MC or data
    y_max = max(
        np.max(total_yield),
        np.max([h["yields"] for h in histogram_dict_list if h["isData"]]),
    )
    # lowest MC yield in single bin (not considering empty bins)
    y_min = np.min(total_yield[np.nonzero(total_yield)])

    # determine scale setting, unless it is provided
    if log_scale is None:
        # if yields vary over more than 2 orders of magnitude, set y-axis to log scale
        log_scale = (y_max / y_min) > 100

    # set vertical axis scale and limits
    if log_scale:
        # use log scale
        ax1.set_yscale("log")
        ax1.set_ylim([y_min / 10, y_max * 10])
        # add "_log" to the figure name
        figure_path = figure_path.with_name(
            figure_path.stem + "_log" + figure_path.suffix
        )
    else:
        # do not use log scale
        ax1.set_ylim([0, y_max * 1.5])  # 50% headroom

    # MC contributions in inverse order, such that first legend entry corresponds to
    # the last (highest) contribution to the stack
    all_containers = mc_containers[::-1] + [mc_unc_container, data_container]
    all_labels = mc_labels[::-1] + ["Uncertainty", data_label]
    ax1.legend(all_containers, all_labels, frameon=False, fontsize="large")

    ax1.set_xlim(bin_left_edges[0], bin_right_edges[-1])
    ax1.set_ylabel("events")
    ax1.set_xticklabels([])
    ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
    ax1.tick_params(direction="in", top=True, right=True, which="both")

    ax2.set_xlim(bin_left_edges[0], bin_right_edges[-1])
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel(histogram_dict_list[0]["variable"])
    ax2.set_ylabel("data / model")
    ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")

    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def correlation_matrix(
    corr_mat: np.ndarray,
    labels: Union[List[str], np.ndarray],
    figure_path: pathlib.Path,
) -> None:
    """Draws a correlation matrix.

    Args:
        corr_mat (np.ndarray): the correlation matrix to plot
        labels (Union[List[str], np.ndarray]): names of parameters in the correlation
            matrix
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

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def pulls(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: Union[List[str], np.ndarray],
    figure_path: pathlib.Path,
) -> None:
    """Draws a pull plot.

    Args:
        bestfit (np.ndarray): best-fit parameter results
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
    ax.set_yticklabels(labels)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())  # minor ticks
    ax.tick_params(axis="both", which="major", pad=8)
    ax.tick_params(direction="in", top=True, right=True, which="both")
    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


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
        impact_prefit_down (np.ndarray): pre-fit impact in "down" direction per
            parameter
        impact_postfit_up (np.ndarray): post-fit impact in "up" direction per parameter
        impact_postfit_down (np.ndarray): post-fit impact in "down" direction per
            parameter
        figure_path (pathlib.Path): path where figure should be saved
    """
    num_pars = len(bestfit)

    mpl.style.use("seaborn-colorblind")
    fig, ax_pulls = plt.subplots(figsize=(8, 2.5 + num_pars * 0.45), dpi=100)
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

    y_pos = np.arange(num_pars)[::-1]

    # pre-fit up
    pre_up = ax_impact.barh(
        y_pos, impact_prefit_up, fill=False, linewidth=1, edgecolor="C0"
    )
    # pre-fit down
    pre_down = ax_impact.barh(
        y_pos, impact_prefit_down, fill=False, linewidth=1, edgecolor="C5"
    )
    # post-fit up
    post_up = ax_impact.barh(y_pos, impact_postfit_up, color="C0")
    # post-fit down
    post_down = ax_impact.barh(y_pos, impact_postfit_down, color="C5")
    # nuisance parameter pulls
    pulls = ax_pulls.errorbar(bestfit, y_pos, xerr=uncertainty, fmt="o", color="k")

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
    ax_pulls.set_yticklabels(labels)

    ax_pulls.tick_params(direction="in", which="both")
    ax_impact.tick_params(direction="in", which="both")

    fig.legend(
        (pre_up, pre_down, post_up, post_down, pulls),
        (
            r"pre-fit impact: $\theta = \hat{\theta} + \Delta \theta$",
            r"pre-fit impact: $\theta = \hat{\theta} - \Delta \theta$",
            r"post-fit impact: $\theta = \hat{\theta} + \Delta \hat{\theta}$",
            r"post-fit impact: $\theta = \hat{\theta} - \Delta \hat{\theta}$",
            "pulls",
        ),
        frameon=False,
        loc="upper left",
        ncol=3,
        fontsize="large",
    )
    leg_space = 1.0 / (num_pars + 3) + 0.03
    fig.tight_layout(rect=[0, 0, 1.0, 1 - leg_space])  # make space for legend on top

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def templates(
    nominal_histo: Dict[str, np.ndarray],
    up_histo: Dict[str, np.ndarray],
    down_histo: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    variable: str,
    figure_path: pathlib.Path,
) -> None:
    """Draws a nominal template and the associated up/down variations.

    If a variation template is an empty dict, it is not drawn.

    Args:
        nominal_histo (Dict[str, np.ndarray]): the nominal template
        up_histo (Dict[str, np.ndarray]): the "up" variation
        down_histo (Dict[str, np.ndarray]): the "down" variation
        bin_edges (np.ndarray): bin edges of histogram
        variable (str): variable name for the horizontal axis
        figure_path (pathlib.Path): path where figure should be saved
    """
    bin_width = bin_edges[1:] - bin_edges[:-1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mpl.style.use("seaborn-colorblind")
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ratio plot line through unity and stat. uncertainty of nominal
    ax2.plot(
        [bin_edges[0], bin_edges[-1]],
        [1, 1],
        "--",
        color="black",
        linewidth=1,
    )
    rel_nominal_stat_unc = nominal_histo["stdev"] / nominal_histo["yields"]
    ax2.bar(
        bin_centers,
        2 * rel_nominal_stat_unc,
        width=bin_width,
        bottom=1 - rel_nominal_stat_unc,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    colors = ["black", "C0", "C1"]
    linestyles = ["-", "--", "-."]
    all_templates = [nominal_histo, up_histo, down_histo]
    labels = ["nominal", "up", "down"]

    # x positions for lines drawn showing the template distributions
    line_x = [y for y in bin_edges for _ in range(2)][1:-1]

    # draw templates
    for template, color, linestyle, label in zip(
        all_templates, colors, linestyles, labels
    ):
        if not template:
            # variation not defined
            continue

        # lines to show each template distribution
        line_y = [y for y in template["yields"] for _ in range(2)]

        ax1.plot(
            line_x,
            line_y,
            "-",
            color=color,
            linestyle=linestyle,
            label=label,
        )
        if label == "nominal":
            # band for stat. uncertainty of nominal prediction
            ax1.bar(
                bin_centers,
                2 * nominal_histo["stdev"],
                width=bin_width,
                bottom=nominal_histo["yields"] - nominal_histo["stdev"],
                fill=False,
                linewidth=0,
                edgecolor="gray",
                hatch=3 * "/",
            )
        else:
            # error bars for up/down variations
            ax1.errorbar(
                bin_centers,
                template["yields"],
                yerr=template["stdev"],
                fmt="none",
                color=color,
            )

            # ratio plot: variation / nominal
            template_ratio_plot = template["yields"] / nominal_histo["yields"]
            line_y = [y for y in template_ratio_plot for _ in range(2)]

            ax2.plot(
                line_x,
                line_y,
                "-",
                color=color,
                linestyle=linestyle,
            )
            ax2.errorbar(
                bin_centers,
                template_ratio_plot,
                yerr=template["stdev"] / nominal_histo["yields"],
                fmt="none",
                color=color,
            )

    # increase font sizes
    for item in (
        [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label]
        + ax1.get_yticklabels()
        + ax2.get_xticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks on all axes
    for axis in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax1.legend(frameon=False, fontsize="large")

    max_yield = max([max(template["yields"]) for template in all_templates if template])

    ax1.set_xlim([bin_edges[0], bin_edges[-1]])
    ax1.set_ylim([0, max_yield * 1.5])
    ax1.set_ylabel("events")
    ax1.set_xticklabels([])
    ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
    ax1.tick_params(direction="in", top=True, right=True, which="both")

    ax2.set_xlim([bin_edges[0], bin_edges[-1]])
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel(variable)
    ax2.set_ylabel("variation / nominal")
    ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")

    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def scan(
    par_name: str,
    par_mle: float,
    par_unc: float,
    par_vals: np.ndarray,
    par_nlls: np.ndarray,
    figure_path: pathlib.Path,
) -> None:
    """Draws a figure showing the results of a likelihood scan.

    Args:
        par_name (str): name of parameter used in scan
        par_mle (float): best-fit result for parameter
        par_unc (float): best-fit parameter uncertainty
        par_vals (np.ndarray): values used in scan over parameter
        par_nlls (np.ndarray): -2 log(L) offset at each scan point
        figure_path (pathlib.Path): path where figure should be saved
    """
    mpl.style.use("seaborn-colorblind")
    fig, ax = plt.subplots()

    # line through y=1 and y=4 to show confidence intervals
    ax.plot([par_vals[0], par_vals[-1]], [1, 1], ":", color="gray")
    ax.plot([par_vals[0], par_vals[-1]], [4, 4], ":", color="gray")

    # position for text - right edge of the figure, with slight padding
    text_x_pos = par_vals[-1] - 0.01 * (par_vals[-1] - par_vals[0])
    ax.text(
        text_x_pos,
        1.0,
        "68% CL",
        horizontalalignment="right",
        verticalalignment="bottom",
        color="gray",
    )
    ax.text(
        text_x_pos,
        4.0,
        "95% CL",
        horizontalalignment="right",
        verticalalignment="bottom",
        color="gray",
    )

    # Gaussian at best-fit parameter value for reference
    val_grid = np.linspace(par_vals[0], par_vals[-1], 100)
    gaussian_approx = [((par_val - par_mle) / par_unc) ** 2 for par_val in val_grid]
    ax.plot(val_grid, gaussian_approx, "--", color="C5", label="Gaussian approximation")

    # scan results
    ax.plot(par_vals, par_nlls, "-", color="C0")
    ax.plot(par_vals, par_nlls, "X", color="C0", label="parameter scan")

    # increase font sizes
    for item in (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.set_xlabel(par_name)
    ax.set_xlim(par_vals[0], par_vals[-1])
    ax.set_ylabel(r"$-2 \Delta \log(L)$")
    ax.set_ylim(0, max(par_nlls) * 1.2)
    ax.tick_params(axis="both", which="major", pad=8)
    ax.tick_params(direction="in", top=True, right=True, which="both")

    ax.legend(frameon=False, fontsize="large")

    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)


def limit(
    observed_CLs: np.ndarray,
    expected_CLs: np.ndarray,
    poi_values: np.ndarray,
    figure_path: pathlib.Path,
) -> None:
    """Draws observed and expected CLs values as function of the parameter of interest.

    Args:
        observed_CLs (np.ndarray): observed CLs values
        expected_CLs (np.ndarray): expected CLs values, including 1 and 2 sigma bands
        poi_values (np.ndarray): parameter of interest values used in scan
        figure_path (pathlib.Path): path where figure should be saved
    """
    fig, ax = plt.subplots()

    xmin = min(poi_values)
    xmax = max(poi_values)

    # line through CLs = 0.05
    ax.hlines(
        0.05,
        xmin=xmin,
        xmax=xmax,
        linestyle="dashdot",
        color="red",
        label=r"CL$_S$ = 5%",
    )

    # 1 and 2 sigma bands
    ax.fill_between(
        poi_values,
        expected_CLs[:, 0],
        expected_CLs[:, 4],
        color="yellow",
        label=r"expected CL$_S$ $\pm 2\sigma$",
    )
    ax.fill_between(
        poi_values,
        expected_CLs[:, 1],
        expected_CLs[:, 3],
        color="limegreen",
        label=r"expected CL$_S$ $\pm 1\sigma$",
    )

    # expected CLs
    ax.plot(
        poi_values,
        expected_CLs[:, 2],
        "--",
        color="black",
        label=r"expected CL$_S$",
    )

    # observed CLs values
    ax.plot(poi_values, observed_CLs, "o-", color="black", label=r"observed CL$_S$")

    # increase font sizes
    for item in (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.legend(frameon=False, fontsize="large")

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\mathrm{CL}_{s}$")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 1])
    ax.tick_params(axis="both", which="major", pad=8)
    ax.tick_params(direction="in", top=True, right=True, which="both")

    fig.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {figure_path}")
    fig.savefig(figure_path)
    plt.close(fig)
