"""Visualizes inference results with matplotlib."""

import logging
import math
import pathlib
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import packaging.version

from cabinetry.visualize import utils


log = logging.getLogger(__name__)

# handling of matplotlib<3.6 (for Python 3.7)
if packaging.version.parse(mpl.__version__) < packaging.version.parse("3.6"):
    MPL_GEQ_36 = False  # pragma: no cover
    MPL_STYLE = "seaborn-colorblind"  # pragma: no cover
else:
    MPL_GEQ_36 = True
    MPL_STYLE = "seaborn-v0_8-colorblind"


def correlation_matrix(
    corr_mat: np.ndarray,
    labels: Union[List[str], np.ndarray],
    *,
    figure_path: Optional[pathlib.Path] = None,
    close_figure: bool = False,
) -> mpl.figure.Figure:
    """Draws a correlation matrix.

    Args:
        corr_mat (np.ndarray): the correlation matrix to plot
        labels (Union[List[str], np.ndarray]): names of parameters in the correlation
            matrix
        figure_path (Optional[pathlib.Path], optional): path where figure should be
            saved, or None to not save it, defaults to None
        close_figure (bool, optional): whether to close each figure immediately after
            saving it, defaults to False (enable when producing many figures to avoid
            memory issues, prevents rendering in notebooks)

    Returns:
        matplotlib.figure.Figure: the correlation matrix figure
    """
    # rounding for test in CI to match reference
    fig, ax = plt.subplots(
        figsize=(round(5 + len(labels) / 1.6, 1), round(3 + len(labels) / 1.6, 1)),
        dpi=100,
        layout="constrained",
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

    # add correlation as text
    for (j, i), corr in np.ndenumerate(corr_mat):
        text_color = "white" if abs(corr_mat[j, i]) > 0.75 else "black"
        if abs(corr) > 0.005:
            ax.text(i, j, f"{corr:.2f}", ha="center", va="center", color=text_color)

    utils._save_and_close(fig, figure_path, close_figure)
    return fig


def pulls(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: Union[List[str], np.ndarray],
    *,
    figure_path: Optional[pathlib.Path] = None,
    close_figure: bool = False,
) -> mpl.figure.Figure:
    """Draws a pull plot.

    Args:
        bestfit (np.ndarray): best-fit parameter results
        uncertainty (np.ndarray): parameter uncertainties
        labels (Union[List[str], np.ndarray]): parameter names
        figure_path (Optional[pathlib.Path], optional): path where figure should be
            saved, or None to not save it, defaults to None
        close_figure (bool, optional): whether to close each figure immediately after
            saving it, defaults to False (enable when producing many figures to avoid
            memory issues, prevents rendering in notebooks)

    Returns:
        matplotlib.figure.Figure: the pull figure
    """
    num_pars = len(bestfit)
    y_positions = np.arange(num_pars)[::-1]
    fig, ax = plt.subplots(figsize=(6, 1 + num_pars / 4), dpi=100, layout="constrained")
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

    utils._save_and_close(fig, figure_path, close_figure)
    return fig


def ranking(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: Union[List[str], np.ndarray],
    impact_prefit_up: np.ndarray,
    impact_prefit_down: np.ndarray,
    impact_postfit_up: np.ndarray,
    impact_postfit_down: np.ndarray,
    *,
    figure_path: Optional[pathlib.Path] = None,
    close_figure: bool = False,
) -> mpl.figure.Figure:
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
        figure_path (Optional[pathlib.Path], optional): path where figure should be
            saved, or None to not save it, defaults to None
        close_figure (bool, optional): whether to close each figure immediately after
            saving it, defaults to False (enable when producing many figures to avoid
            memory issues, prevents rendering in notebooks)

    Returns:
        matplotlib.figure.Figure: the ranking figure
    """
    num_pars = len(bestfit)

    # layout to make space for legend on top
    leg_space = 1.0 / (num_pars + 3) + 0.03
    if MPL_GEQ_36:
        import matplotlib.layout_engine

        layout = matplotlib.layout_engine.ConstrainedLayoutEngine(
            rect=[0, 0, 1.0, 1 - leg_space]
        )
    else:
        layout = None  # pragma: no cover  # layout set after figure creation instead

    mpl.style.use(MPL_STYLE)
    fig, ax_pulls = plt.subplots(
        figsize=(8, 2.5 + num_pars * 0.45), dpi=100, layout=layout
    )
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
    impact_max = np.amax(np.fabs(np.hstack((impact_prefit_up, impact_prefit_down))))
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

    if not MPL_GEQ_36:
        fig.tight_layout(rect=[0, 0, 1.0, 1 - leg_space])  # pragma: no cover

    utils._save_and_close(fig, figure_path, close_figure)
    return fig


def scan(
    par_name: str,
    par_mle: float,
    par_unc: float,
    par_vals: np.ndarray,
    par_nlls: np.ndarray,
    *,
    figure_path: Optional[pathlib.Path] = None,
    close_figure: bool = False,
) -> mpl.figure.Figure:
    """Draws a figure showing the results of a likelihood scan.

    Args:
        par_name (str): name of parameter used in scan
        par_mle (float): best-fit result for parameter
        par_unc (float): best-fit parameter uncertainty
        par_vals (np.ndarray): values used in scan over parameter
        par_nlls (np.ndarray): -2 log(L) offset at each scan point
        figure_path (Optional[pathlib.Path], optional): path where figure should be
            saved, or None to not save it, defaults to None
        close_figure (bool, optional): whether to close each figure immediately after
            saving it, defaults to False (enable when producing many figures to avoid
            memory issues, prevents rendering in notebooks)

    Returns:
        matplotlib.figure.Figure: the likelihood scan figure
    """
    mpl.style.use(MPL_STYLE)
    fig, ax = plt.subplots(layout="constrained")

    y_lim = max(par_nlls) * 1.2  # upper y-axis limit, 20% headroom

    # lines through y=1 and y=4 to show confidence intervals
    ax.plot([par_vals[0], par_vals[-1]], [1, 1], ":", color="gray")
    ax.plot([par_vals[0], par_vals[-1]], [4, 4], ":", color="gray")

    # position for text - right edge of the figure, with slight padding
    text_x_pos = par_vals[-1] - 0.01 * (par_vals[-1] - par_vals[0])
    # only draw text if it fits in the figure
    if y_lim >= 1:
        ax.text(text_x_pos, 1.0, "68% CL", ha="right", va="bottom", color="gray")
    if y_lim >= 4:
        ax.text(text_x_pos, 4.0, "95% CL", ha="right", va="bottom", color="gray")

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
    ax.set_ylim(0, y_lim)
    ax.tick_params(axis="both", which="major", pad=8)
    ax.tick_params(direction="in", top=True, right=True, which="both")

    ax.legend(frameon=False, fontsize="large")

    utils._save_and_close(fig, figure_path, close_figure)
    return fig


def limit(
    observed_CLs: np.ndarray,
    expected_CLs: np.ndarray,
    poi_values: np.ndarray,
    cls_target: float,
    *,
    figure_path: Optional[pathlib.Path] = None,
    close_figure: bool = False,
) -> mpl.figure.Figure:
    """Draws observed and expected CLs values as function of the parameter of interest.

    Args:
        observed_CLs (np.ndarray): observed CLs values
        expected_CLs (np.ndarray): expected CLs values, including 1 and 2 sigma bands
        poi_values (np.ndarray): parameter of interest values used in scan
        cls_target (float): target CLs value to visualize as horizontal line
        figure_path (Optional[pathlib.Path], optional): path where figure should be
            saved, or None to not save it, defaults to None
        close_figure (bool, optional): whether to close each figure immediately after
            saving it, defaults to False (enable when producing many figures to avoid
            memory issues, prevents rendering in notebooks)

    Returns:
        matplotlib.figure.Figure: the CLs figure
    """
    fig, ax = plt.subplots(layout="constrained")

    xmin = min(poi_values)
    xmax = max(poi_values)

    # observed CLs values
    ax.plot(poi_values, observed_CLs, "o-", color="black", label=r"observed CL$_S$")

    # expected CLs
    ax.plot(
        poi_values, expected_CLs[:, 2], "--", color="black", label=r"expected CL$_S$"
    )

    # 1 and 2 sigma bands
    ax.fill_between(
        poi_values,
        expected_CLs[:, 1],
        expected_CLs[:, 3],
        color="limegreen",
        label=r"expected CL$_S$ $\pm 1\sigma$",
    )
    ax.fill_between(
        poi_values,
        expected_CLs[:, 0],
        expected_CLs[:, 4],
        color="yellow",
        label=r"expected CL$_S$ $\pm 2\sigma$",
        zorder=0,  # draw beneath 1 sigma band
    )

    # determine whether CLs value shown in percent is integer (float.is_integer() is not
    # sufficient after calculation of 1-confidence_level in fit.limit)
    cls_pct_is_integer = math.isclose(cls_target * 100, round(cls_target * 100))
    cls_label = f"{cls_target:.{0 if cls_pct_is_integer else 2}%}"
    # line through CLs = cls_target
    ax.hlines(
        cls_target,
        xmin=xmin,
        xmax=xmax,
        linestyle="dashdot",
        color="red",
        label="CL$_S$ = " + cls_label,  # 2 decimals unless they are both 0, then 0
        zorder=1,  # draw beneath observed / expected
    )

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

    utils._save_and_close(fig, figure_path, close_figure)
    return fig
