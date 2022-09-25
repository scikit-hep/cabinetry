from unittest import mock

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import numpy as np

from cabinetry.visualize import plot_result


def test_correlation_matrix(tmp_path):
    fname = tmp_path / "fig.png"
    # one parameter is below threshold so no text is shown for it on the plot
    corr_mat = np.asarray([[1.0, 0.35, 0.002], [0.35, 1.0, -0.2], [0.002, -0.2, 1.0]])
    labels = ["a", "b", "c"]

    fig = plot_result.correlation_matrix(corr_mat, labels, figure_path=fname)
    assert (
        compare_images(
            "tests/visualize/reference/correlation_matrix.png", str(fname), 0
        )
        is None
    )

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"

    # adjust layout behavior, see https://github.com/matplotlib/matplotlib/issues/21742
    if plot_result.MPL_GEQ_36:
        fig.set_layout_engine(None)
    else:
        fig.set_constrained_layout(False)

    fig.savefig(fname)
    assert (
        compare_images(
            "tests/visualize/reference/correlation_matrix.png", str(fname), 0
        )
        is None
    )

    # do not save figure, but close it
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = plot_result.correlation_matrix(corr_mat, labels, close_figure=True)
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")


def test_pulls(tmp_path):
    fname = tmp_path / "fig.png"
    bestfit = np.asarray([-0.2, 0.0, 0.1])
    uncertainty = np.asarray([0.9, 1.0, 0.7])
    labels = np.asarray(["a", "b", "c"])

    fig = plot_result.pulls(bestfit, uncertainty, labels, figure_path=fname)
    assert compare_images("tests/visualize/reference/pulls.png", str(fname), 0) is None

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert compare_images("tests/visualize/reference/pulls.png", str(fname), 0) is None

    # do not save figure, but close it
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = plot_result.pulls(bestfit, uncertainty, labels, close_figure=True)
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")


def test_ranking(tmp_path):
    fname = tmp_path / "fig.png"
    bestfit = np.asarray([0.3, -0.1])
    uncertainty = np.asarray([0.8, 1.0])
    labels = np.asarray(["jet energy scale", "modeling uncertainty"])
    impact_prefit_up = np.asarray([0.5, 0.3])
    impact_prefit_down = np.asarray([-0.4, -0.3])
    impact_postfit_up = np.asarray([0.4, 0.25])
    impact_postfit_down = np.asarray([-0.3, -0.25])

    fig = plot_result.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        figure_path=fname,
    )
    # large tolerance needed here, possibly related to lack of set_tight_layout usage
    assert (
        compare_images("tests/visualize/reference/ranking.png", str(fname), 50) is None
    )

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert (
        compare_images("tests/visualize/reference/ranking.png", str(fname), 50) is None
    )

    # do not save figure, but close it
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = plot_result.ranking(
            bestfit,
            uncertainty,
            labels,
            impact_prefit_up,
            impact_prefit_down,
            impact_postfit_up,
            impact_postfit_down,
            close_figure=True,
        )
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")


def test_scan(tmp_path):
    fname = tmp_path / "fig.png"
    par_name = "a"
    par_mle = 1.5
    par_unc = 0.2
    par_vals = np.asarray([1.1, 1.3, 1.5, 1.7, 1.9])
    par_nlls = np.asarray([4.1, 1.0, 0.0, 1.1, 3.9])

    fig = plot_result.scan(
        par_name, par_mle, par_unc, par_vals, par_nlls, figure_path=fname
    )
    assert compare_images("tests/visualize/reference/scan.png", str(fname), 0) is None

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert compare_images("tests/visualize/reference/scan.png", str(fname), 0) is None

    # do not save figure, but close it
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        # no 68% CL / 95% CL text
        par_nlls = np.asarray([0.1, 0.04, 0.0, 0.04, 0.1])
        fig = plot_result.scan(
            par_name, par_mle, par_unc, par_vals, par_nlls, close_figure=True
        )
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")


def test_limit(tmp_path):
    fname = tmp_path / "fig.png"
    observed_CLs = np.asarray([0.31, 0.05, 0.005, 0.0001])
    expected_CLs = np.asarray(
        [
            [0.07, 0.16, 0.35, 0.62, 0.88],
            [0.0, 0.01, 0.06, 0.23, 0.56],
            [0.0, 0.0, 0.0, 0.05, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.04],
        ]
    )
    poi_values = np.asarray([0.5, 1.0, 1.5, 2.0])
    cls_target = 0.05

    fig = plot_result.limit(
        observed_CLs, expected_CLs, poi_values, cls_target, figure_path=fname
    )
    assert compare_images("tests/visualize/reference/limit.png", str(fname), 0) is None

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert compare_images("tests/visualize/reference/limit.png", str(fname), 0) is None

    # do not save figure, but close it
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = plot_result.limit(
            observed_CLs, expected_CLs, poi_values, cls_target, close_figure=True
        )
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")
