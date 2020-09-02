from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from cabinetry.contrib import matplotlib_visualize


def test__total_yield_uncertainty():
    stdev_list = [np.asarray([0.1, 0.2, 0.1]), np.asarray([0.3, 0.2, 0.1])]
    expected_uncertainties = [0.31622777, 0.28284271, 0.14142136]
    assert np.allclose(
        matplotlib_visualize._total_yield_uncertainty(stdev_list),
        expected_uncertainties,
    )


@pytest.mark.parametrize("figure_extension", ["fig.pdf", "subdir/fig.pdf"])
def test_data_MC(figure_extension, tmp_path):
    fname = tmp_path / figure_extension
    bg_hist = {
        "yields": np.asarray([12.5, 14]),
        "stdev": np.asarray([0.4, 0.5]),
        "bins": np.asarray([1, 2, 3]),
    }
    sig_hist = {
        "yields": np.asarray([2, 5]),
        "stdev": np.asarray([0.1, 0.2]),
        "bins": np.asarray([1, 2, 3]),
    }
    data_hist = {
        "yields": np.asarray([13, 15]),
        "stdev": np.asarray([3.61, 3.87]),
        "bins": np.asarray([1, 2, 3]),
    }
    histo_dict_list = [
        {"label": "Background", "isData": False, "hist": bg_hist, "variable": "x"},
        {"label": "Signal", "isData": False, "hist": sig_hist, "variable": "x"},
        {"label": "Data", "isData": True, "hist": data_hist, "variable": "x"},
    ]
    matplotlib_visualize.data_MC(histo_dict_list, fname)
    assert compare_images("tests/contrib/reference/data_MC.pdf", str(fname), 0) is None


@pytest.mark.parametrize("figure_extension", ["fig.pdf", "subdir/fig.pdf"])
def test_correlation_matrix(figure_extension, tmp_path):
    fname = tmp_path / figure_extension
    # one parameter is below threshold so no text is shown for it on the plot
    corr_mat = np.asarray([[1.0, 0.35, 0.002], [0.35, 1.0, -0.2], [0.002, -0.2, 1.0]])
    labels = ["a", "b", "c"]
    matplotlib_visualize.correlation_matrix(corr_mat, labels, fname)
    assert (
        compare_images("tests/contrib/reference/correlation_matrix.pdf", str(fname), 0)
        is None
    )


@pytest.mark.parametrize("figure_extension", ["fig.pdf", "subdir/fig.pdf"])
def test_pulls(figure_extension, tmp_path):
    fname = tmp_path / figure_extension
    bestfit = np.asarray([-0.2, 0.0, 0.1])
    uncertainty = np.asarray([0.9, 1.0, 0.7])
    labels = np.asarray(["a", "b", "c"])
    matplotlib_visualize.pulls(bestfit, uncertainty, labels, fname)
    assert compare_images("tests/contrib/reference/pulls.pdf", str(fname), 0) is None


@pytest.mark.parametrize("figure_extension", ["fig.pdf", "subdir/fig.pdf"])
def test_ranking(figure_extension, tmp_path):
    fname = tmp_path / figure_extension
    bestfit = np.asarray([0.3, -0.1])
    uncertainty = np.asarray([0.8, 1.0])
    labels = np.asarray(["jet energy scale", "modeling uncertainty"])
    impact_prefit_up = np.asarray([0.5, 0.3])
    impact_prefit_down = np.asarray([-0.4, -0.3])
    impact_postfit_up = np.asarray([0.4, 0.25])
    impact_postfit_down = np.asarray([-0.3, -0.25])

    matplotlib_visualize.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        fname,
    )
    assert compare_images("tests/contrib/reference/ranking.pdf", str(fname), 0) is None


def test_templates(tmp_path):
    fname = tmp_path / "fig.pdf"
    nominal_histo = {
        "yields": np.asarray([1.0, 1.2]),
        "stdev": np.asarray([0.05, 0.06]),
    }
    up_histo = {"yields": np.asarray([1.1, 1.4]), "stdev": np.asarray([0.05, 0.07])}
    down_histo = {"yields": np.asarray([0.8, 0.9]), "stdev": np.asarray([0.06, 0.07])}
    bin_edges = np.asarray([0.0, 1.0, 2.0])
    variable = "x"

    matplotlib_visualize.templates(
        nominal_histo,
        up_histo,
        down_histo,
        bin_edges,
        variable,
        fname,
    )
    assert (
        compare_images("tests/contrib/reference/templates.pdf", str(fname), 0) is None
    )
