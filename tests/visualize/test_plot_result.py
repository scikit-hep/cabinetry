import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import numpy as np

from cabinetry.visualize import plot_result


def test_no_open_figure():
    # ensure there are no open figures at the start, if this fails then some other part
    # of the test suite opened a figure without closing it
    assert len(plt.get_fignums()) == 0


def test_correlation_matrix(tmp_path):
    fname = tmp_path / "fig.pdf"
    # one parameter is below threshold so no text is shown for it on the plot
    corr_mat = np.asarray([[1.0, 0.35, 0.002], [0.35, 1.0, -0.2], [0.002, -0.2, 1.0]])
    labels = ["a", "b", "c"]
    plot_result.correlation_matrix(corr_mat, labels, fname)
    assert (
        compare_images(
            "tests/visualize/reference/correlation_matrix.pdf", str(fname), 0
        )
        is None
    )

    # single open figure, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 1
    plot_result.correlation_matrix(corr_mat, labels, fname, close_figure=True)
    assert len(plt.get_fignums()) == 1
    plt.close("all")


def test_pulls(tmp_path):
    fname = tmp_path / "fig.pdf"
    bestfit = np.asarray([-0.2, 0.0, 0.1])
    uncertainty = np.asarray([0.9, 1.0, 0.7])
    labels = np.asarray(["a", "b", "c"])
    plot_result.pulls(bestfit, uncertainty, labels, fname)
    assert compare_images("tests/visualize/reference/pulls.pdf", str(fname), 0) is None

    # single open figure, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 1
    plot_result.pulls(bestfit, uncertainty, labels, fname, close_figure=True)
    assert len(plt.get_fignums()) == 1
    plt.close("all")


def test_ranking(tmp_path):
    fname = tmp_path / "fig.pdf"
    bestfit = np.asarray([0.3, -0.1])
    uncertainty = np.asarray([0.8, 1.0])
    labels = np.asarray(["jet energy scale", "modeling uncertainty"])
    impact_prefit_up = np.asarray([0.5, 0.3])
    impact_prefit_down = np.asarray([-0.4, -0.3])
    impact_postfit_up = np.asarray([0.4, 0.25])
    impact_postfit_down = np.asarray([-0.3, -0.25])

    plot_result.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        fname,
    )
    assert (
        compare_images("tests/visualize/reference/ranking.pdf", str(fname), 0) is None
    )

    # single open figure, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 1
    plot_result.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        fname,
        close_figure=True,
    )
    assert len(plt.get_fignums()) == 1
    plt.close("all")


def test_scan(tmp_path):
    fname = tmp_path / "fig.pdf"
    par_name = "a"
    par_mle = 1.5
    par_unc = 0.2
    par_vals = np.asarray([1.1, 1.3, 1.5, 1.7, 1.9])
    par_nlls = np.asarray([4.1, 1.0, 0.0, 1.1, 3.9])

    plot_result.scan(par_name, par_mle, par_unc, par_vals, par_nlls, fname)
    assert compare_images("tests/visualize/reference/scan.pdf", str(fname), 0) is None

    # single open figure, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 1

    # no 68% CL / 95% CL text
    par_nlls = np.asarray([0.1, 0.04, 0.0, 0.04, 0.1])
    plot_result.scan(
        par_name, par_mle, par_unc, par_vals, par_nlls, fname, close_figure=True
    )
    assert len(plt.get_fignums()) == 1
    plt.close("all")


def test_limit(tmp_path):
    fname = tmp_path / "fig.pdf"
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

    plot_result.limit(observed_CLs, expected_CLs, poi_values, fname)
    assert compare_images("tests/visualize/reference/limit.pdf", str(fname), 0) is None

    # single open figure, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 1
    plot_result.limit(observed_CLs, expected_CLs, poi_values, fname, close_figure=True)
    assert len(plt.get_fignums()) == 1
    plt.close("all")
