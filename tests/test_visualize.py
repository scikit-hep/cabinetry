from collections import namedtuple
import pathlib
from unittest import mock

import numpy as np
import pytest

from cabinetry import fit
from cabinetry import visualize


MockHistogram = namedtuple("MockHistogram", ["bins", "yields", "stdev"])


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("SR", True), "SR_prefit.pdf"),
        (("SR", False), "SR_postfit.pdf"),
        (("SR 1", True), "SR-1_prefit.pdf"),
        (("SR 1", False), "SR-1_postfit.pdf"),
    ],
)
def test__build_figure_name(test_input, expected):
    assert visualize._build_figure_name(*test_input) == expected


def test__total_yield_uncertainty():
    stdev_list = [np.asarray([0.1, 0.2, 0.1]), np.asarray([0.3, 0.2, 0.1])]
    expected_uncertainties = [0.31622777, 0.28284271, 0.14142136]
    assert np.allclose(
        visualize._total_yield_uncertainty(stdev_list), expected_uncertainties,
    )


@mock.patch("cabinetry.visualize._total_yield_uncertainty", return_value=[0.2])
@mock.patch("cabinetry.contrib.matplotlib_visualize.data_MC")
@mock.patch(
    "cabinetry.histo.Histogram.from_config",
    return_value=MockHistogram([0.0, 1.0], [1.0], [0.1]),
)
def test_data_MC_from_histograms(mock_load, mock_draw, mock_stdev, tmp_path):
    """contrib.matplotlib_visualize is only imported depending on the keyword argument,
    so cannot patch via cabinetry.visualize.matplotlib_visualize
    Generally it seems like following the path to the module is preferred, but that
    does not work for the ``data_MC`` case. For some information see also
    https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    """
    config = {
        "General": {"HistogramFolder": tmp_path},
        "Regions": [{"Name": "reg_1", "Variable": "x"}],
        "Samples": [{"Name": "sample_1"}, {"Name": "data", "Data": True}],
    }

    visualize.data_MC_from_histograms(config, tmp_path, method="matplotlib")

    # the call_args_list contains calls (outer round brackets), first filled with
    # arguments (inner round brackets) and then keyword arguments
    assert mock_load.call_args_list == [
        (
            (
                tmp_path,
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "sample_1"},
                {"Name": "nominal"},
            ),
            {"modified": True},
        ),
        (
            (
                tmp_path,
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "data", "Data": True},
                {"Name": "nominal"},
            ),
            {"modified": True},
        ),
    ]
    assert mock_stdev.call_args_list == [(([[0.1]],), {})]
    assert mock_draw.call_args_list == [
        (
            (
                [
                    {
                        "label": "sample_1",
                        "isData": False,
                        "hist": {"yields": [1.0]},
                        "variable": "x",
                    },
                    {
                        "label": "data",
                        "isData": True,
                        "hist": {"yields": [1.0]},
                        "variable": "x",
                    },
                ],
                [0.2],
                [0.0, 1.0],
                tmp_path / "reg_1_prefit.pdf",
            ),
        )
    ]

    # other plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.data_MC_from_histograms(config, tmp_path, method="unknown")


def test_data_MC():
    ...


@mock.patch("cabinetry.contrib.matplotlib_visualize.correlation_matrix")
def test_correlation_matrix(mock_draw):
    corr_mat = np.asarray([[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.1, 0.1, 1.0]])
    corr_mat_pruned = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    labels = ["a", "b", "c"]
    labels_pruned = ["a", "b"]
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "correlation_matrix.pdf"
    fit_results = fit.FitResults(np.empty(0), np.empty(0), labels, corr_mat, 1.0)

    visualize.correlation_matrix(
        fit_results, folder_path, pruning_threshold=0.15, method="matplotlib"
    )

    mock_draw.assert_called_once()
    assert np.allclose(mock_draw.call_args[0][0], corr_mat_pruned)
    assert np.any(
        [mock_draw.call_args[0][1][i] == labels[i] for i in range(len(labels_pruned))]
    )
    assert mock_draw.call_args[0][2] == figure_path
    assert mock_draw.call_args[1] == {}

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.correlation_matrix(fit_results, folder_path, method="unknown")


@mock.patch("cabinetry.contrib.matplotlib_visualize.pulls")
def test_pulls(mock_draw):
    bestfit = np.asarray([0.8, 1.0, 1.05, 1.1])
    uncertainty = np.asarray([0.9, 1.0, 0.03, 0.7])
    labels = ["a", "b", "staterror_region[bin_0]", "c"]
    exclude_list = ["a"]
    folder_path = "tmp"
    fit_results = fit.FitResults(bestfit, uncertainty, labels, np.empty(0), 1.0)

    filtered_bestfit = np.asarray([1.0, 1.1])
    filtered_uncertainty = np.asarray([1.0, 0.7])
    filtered_labels = np.asarray(["b", "c"])
    figure_path = pathlib.Path(folder_path) / "pulls.pdf"

    # with filtering
    visualize.pulls(
        fit_results, folder_path, exclude_list=exclude_list, method="matplotlib",
    )

    mock_draw.assert_called_once()
    assert np.allclose(mock_draw.call_args[0][0], filtered_bestfit)
    assert np.allclose(mock_draw.call_args[0][1], filtered_uncertainty)
    assert np.any(
        [
            mock_draw.call_args[0][2][i] == filtered_labels[i]
            for i in range(len(filtered_labels))
        ]
    )
    assert mock_draw.call_args[0][3] == figure_path
    assert mock_draw.call_args[1] == {}

    # without filtering via list, but with staterror removal
    bestfit_expected = np.asarray([0.8, 1.0, 1.1])
    uncertainty_expected = np.asarray([0.9, 1.0, 0.7])
    labels_expected = ["a", "b", "c"]
    visualize.pulls(fit_results, folder_path, method="matplotlib")

    assert np.allclose(mock_draw.call_args[0][0], bestfit_expected)
    assert np.allclose(mock_draw.call_args[0][1], uncertainty_expected)
    assert np.any(
        [
            mock_draw.call_args[0][2][i] == labels_expected[i]
            for i in range(len(labels_expected))
        ]
    )
    assert mock_draw.call_args[0][3] == figure_path
    assert mock_draw.call_args[1] == {}

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.pulls(
            fit_results, folder_path, exclude_list=exclude_list, method="unknown",
        )


@mock.patch("cabinetry.contrib.matplotlib_visualize.ranking")
def test_ranking(mock_draw):
    bestfit = np.asarray([1.2, 0.1, 0.9])
    uncertainty = np.asarray([0.2, 0.8, 0.5])
    labels = ["staterror_a", "modeling", "mu"]
    impact_prefit_up = np.asarray([0.1, 0.5])
    impact_prefit_down = np.asarray([-0.2, -0.4])
    impact_postfit_up = np.asarray([0.1, 0.4])
    impact_postfit_down = np.asarray([-0.2, -0.3])
    poi_index = 2
    folder_path = "tmp"

    figure_path = pathlib.Path(folder_path) / "ranking.pdf"
    bestfit_expected = np.asarray([0.1, 0.2])
    uncertainty_expected = np.asarray([0.8, 0.2])
    labels_expected = ["modeling", "staterror_a"]

    visualize.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        poi_index,
        folder_path,
    )
    assert mock_draw.call_count == 1
    assert np.allclose(mock_draw.call_args[0][0], bestfit_expected)
    assert np.allclose(mock_draw.call_args[0][1], uncertainty_expected)
    for i_lab, label in enumerate(mock_draw.call_args[0][2]):
        assert label == labels_expected[i_lab]
    assert np.allclose(mock_draw.call_args[0][3], impact_prefit_up[::-1])
    assert np.allclose(mock_draw.call_args[0][4], impact_prefit_down[::-1])
    assert np.allclose(mock_draw.call_args[0][5], impact_postfit_up[::-1])
    assert np.allclose(mock_draw.call_args[0][6], impact_postfit_down[::-1])
    assert mock_draw.call_args[0][7] == figure_path
    assert mock_draw.call_args[1] == {}

    # maximum parameter amount specified
    visualize.ranking(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
        poi_index,
        folder_path,
        max_pars=1,
    )
    assert mock_draw.call_count == 2
    assert np.allclose(mock_draw.call_args[0][0], bestfit_expected[0])
    assert np.allclose(mock_draw.call_args[0][1], uncertainty_expected[0])
    assert mock_draw.call_args[0][2] == labels_expected[0]
    assert np.allclose(mock_draw.call_args[0][3], impact_prefit_up[1])
    assert np.allclose(mock_draw.call_args[0][4], impact_prefit_down[1])
    assert np.allclose(mock_draw.call_args[0][5], impact_postfit_up[1])
    assert np.allclose(mock_draw.call_args[0][6], impact_postfit_down[1])
    assert mock_draw.call_args[0][7] == figure_path
    assert mock_draw.call_args[1] == {}

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.ranking(
            bestfit,
            uncertainty,
            labels,
            impact_prefit_up,
            impact_prefit_down,
            impact_postfit_up,
            impact_postfit_down,
            poi_index,
            folder_path,
            method="unknown",
        )


@mock.patch(
    "cabinetry.histo.Histogram.from_path",
    side_effect=[
        MockHistogram([0.0, 1.0], [2.0], [0.2]),
        MockHistogram([0.0, 1.0], [3.0], [0.3]),
    ]
    * 3,
)
@mock.patch(
    "cabinetry.histo.Histogram.from_config",
    return_value=MockHistogram([0.0, 1.0], [1.0], [0.1]),
)
@mock.patch("cabinetry.contrib.matplotlib_visualize.templates")
def test_templates(mock_draw, mock_histo_config, mock_histo_path, tmp_path):
    # the side effects are repeated for the patched Histogram.from_path
    # to check all relevant behavior (including the unknown backend check)
    nominal_path = tmp_path / "region_sample_nominal_modified.npz"
    up_path = tmp_path / "region_sample_sys_Up_modified.npz"
    down_path = tmp_path / "region_sample_sys_Down_modified.npz"
    region = {"Name": "region", "Variable": "x"}
    sample = {"Name": "sample"}
    config = {
        "General": {"HistogramFolder": tmp_path},
        "Regions": [region],
        "Samples": [sample, {"Name": "data", "Data": True}],
        "Systematics": [{"Name": "sys"}],
    }

    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "templates/region_sample_sys.pdf"

    # add fake histograms for glob
    nominal_path.touch()
    up_path.touch()
    down_path.touch()

    # also add a file that matches pattern but is not needed
    (tmp_path / "region_sample_sys_unknown_modified.npz").touch()

    visualize.templates(config, folder_path)

    assert mock_histo_config.call_args_list == [
        [(tmp_path, region, sample, {"Name": "nominal"}), {}]
    ]
    assert mock_histo_path.call_args_list == [[(down_path,), {}], [(up_path,), {}]]

    nominal = {"yields": [1.0], "stdev": [0.1]}
    up = {"yields": [3.0], "stdev": [0.3]}
    down = {"yields": [2.0], "stdev": [0.2]}
    bins = [0.0, 1.0]
    assert mock_draw.call_args_list == [
        [(nominal, up, down, bins, "x", figure_path), {}]
    ]

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.templates(config, folder_path, method="unknown")

    # remove files for variation histograms
    up_path.unlink()
    down_path.unlink()

    visualize.templates(config, folder_path)
    assert mock_draw.call_count == 1  # no new call, since no variations found
