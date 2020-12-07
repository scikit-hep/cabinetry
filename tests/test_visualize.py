from collections import namedtuple
import pathlib
from unittest import mock

import numpy as np
import pyhf
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
        visualize._total_yield_uncertainty(stdev_list), expected_uncertainties
    )


@mock.patch("cabinetry.visualize._total_yield_uncertainty", return_value=[0.2])
@mock.patch("cabinetry.contrib.matplotlib_visualize.data_MC")
@mock.patch(
    "cabinetry.histo.Histogram.from_config",
    return_value=MockHistogram([0.0, 1.0], [1.0], [0.1]),
)
def test_data_MC_from_histograms(mock_load, mock_draw, mock_stdev):
    """contrib.matplotlib_visualize is only imported depending on the keyword argument,
    so cannot patch via cabinetry.visualize.matplotlib_visualize
    Generally it seems like following the path to the module is preferred, but that
    does not work for the ``data_MC`` case. For some information see also
    https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    """
    config = {
        "General": {"HistogramFolder": "tmp_hist"},
        "Regions": [{"Name": "reg_1", "Variable": "x"}],
        "Samples": [{"Name": "sample_1"}, {"Name": "data", "Data": True}],
    }
    figure_folder = pathlib.Path("tmp")
    histogram_folder = pathlib.Path("tmp_hist")

    visualize.data_MC_from_histograms(
        config, figure_folder=figure_folder, method="matplotlib"
    )

    # the call_args_list contains calls (outer round brackets), first filled with
    # arguments (inner round brackets) and then keyword arguments
    assert mock_load.call_args_list == [
        (
            (
                histogram_folder,
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "data", "Data": True},
                {"Name": "Nominal"},
            ),
            {"modified": True},
        ),
        (
            (
                histogram_folder,
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "sample_1"},
                {"Name": "Nominal"},
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
                        "label": "data",
                        "isData": True,
                        "yields": [1.0],
                        "variable": "x",
                    },
                    {
                        "label": "sample_1",
                        "isData": False,
                        "yields": [1.0],
                        "variable": "x",
                    },
                ],
                [0.2],
                [0.0, 1.0],
                figure_folder / "reg_1_prefit.pdf",
            ),
            {"log_scale": None},
        )
    ]

    # other plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.data_MC_from_histograms(
            config, figure_folder=figure_folder, method="unknown"
        )


@mock.patch("cabinetry.contrib.matplotlib_visualize.data_MC")
@mock.patch("cabinetry.template_builder._get_binning", return_value=np.asarray([1, 2]))
@mock.patch(
    "cabinetry.configuration.get_region_dict",
    return_value={"Name": "region", "Variable": "x"},
)
@mock.patch("cabinetry.tabulate._yields")
@mock.patch("cabinetry.model_utils.calculate_stdev", return_value=[[0.3]])
@mock.patch(
    "cabinetry.model_utils.get_prefit_uncertainties",
    return_value=([0.04956657, 0.0]),
)
@mock.patch(
    "cabinetry.model_utils.get_asimov_parameters",
    return_value=([1.0, 1.0]),
)
def test_data_MC(
    mock_asimov,
    mock_unc,
    mock_stdev,
    mock_table,
    mock_dict,
    mock_bins,
    mock_draw,
    example_spec,
):
    config = {}
    figure_folder = "tmp"
    model_spec = pyhf.Workspace(example_spec).model().spec

    # pre-fit plot
    visualize.data_MC(config, example_spec, figure_folder=figure_folder)

    # Asimov parameter calculation and pre-fit uncertainties
    assert mock_stdev.call_count == 1
    assert mock_asimov.call_args_list[0][0][0].spec == model_spec
    assert mock_unc.call_count == 1
    assert mock_unc.call_args_list[0][0][0].spec == model_spec

    # call to stdev calculation
    assert mock_stdev.call_count == 1
    assert mock_stdev.call_args_list[0][0][0].spec == model_spec
    assert np.allclose(mock_stdev.call_args_list[0][0][1], [1.0, 1.0])
    assert np.allclose(mock_stdev.call_args_list[0][0][2], [0.04956657, 0.0])
    assert np.allclose(
        mock_stdev.call_args_list[0][0][3], np.asarray([[1.0, 0.0], [0.0, 1.0]])
    )
    assert mock_stdev.call_args_list[0][1] == {}

    # yield table
    assert mock_table.call_count == 1
    assert mock_table.call_args_list[0][0][0].spec == model_spec
    assert mock_table.call_args_list[0][0][1] == [np.asarray([[51.839756]])]
    assert mock_table.call_args_list[0][0][2] == [[0.3]]
    assert mock_table.call_args_list[0][0][3] == [np.asarray([475])]
    assert mock_table.call_args_list[0][1] == {}

    assert mock_dict.call_args_list == [[(config, "Signal Region"), {}]]
    assert mock_bins.call_args_list == [[({"Name": "region", "Variable": "x"},), {}]]

    expected_histograms = [
        {
            "label": "Signal",
            "isData": False,
            "yields": np.asarray([51.839756]),
            "variable": "x",
        },
        {
            "label": "Data",
            "isData": True,
            "yields": np.asarray([475]),
            "variable": "x",
        },
    ]
    assert mock_draw.call_count == 1
    assert mock_draw.call_args_list[0][0][0] == expected_histograms
    assert np.allclose(mock_draw.call_args_list[0][0][1], np.asarray([0.3]))
    assert np.allclose(mock_draw.call_args_list[0][0][2], np.asarray([1, 2]))
    assert mock_draw.call_args_list[0][0][3] == pathlib.Path(
        "tmp/Signal-Region_prefit.pdf"
    )
    assert mock_draw.call_args_list[0][1] == {"log_scale": None}

    # post-fit plot and custom scale
    fit_results = fit.FitResults(
        np.asarray([1.01, 1.1]),
        np.asarray([0.03, 0.1]),
        [],
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        0.0,
    )
    visualize.data_MC(
        config,
        example_spec,
        figure_folder=figure_folder,
        fit_results=fit_results,
        log_scale=False,
    )

    assert mock_asimov.call_count == 1  # no new call

    # call to stdev calculation
    assert mock_stdev.call_count == 2
    assert mock_stdev.call_args_list[1][0][0].spec == model_spec
    assert np.allclose(mock_stdev.call_args_list[1][0][1], [1.01, 1.1])
    assert np.allclose(mock_stdev.call_args_list[1][0][2], [0.03, 0.1])
    assert np.allclose(
        mock_stdev.call_args_list[1][0][3], np.asarray([[1.0, 0.2], [0.2, 1.0]])
    )
    assert mock_stdev.call_args_list[1][1] == {}

    assert mock_draw.call_count == 2
    # yield at best-fit point is different from pre-fit
    assert np.allclose(mock_draw.call_args_list[1][0][0][0]["yields"], 57.59396892)
    assert np.allclose(mock_draw.call_args_list[1][0][1], np.asarray([0.3]))
    assert np.allclose(mock_draw.call_args_list[1][0][2], np.asarray([1, 2]))
    assert mock_draw.call_args_list[1][0][3] == pathlib.Path(
        "tmp/Signal-Region_postfit.pdf"
    )
    assert mock_draw.call_args_list[1][1] == {"log_scale": False}

    # no yield table
    visualize.data_MC(config, example_spec, include_table=False)
    assert mock_table.call_count == 2  # 2 calls from before

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.data_MC(
            config, example_spec, figure_folder=figure_folder, method="unknown"
        )


@mock.patch("cabinetry.contrib.matplotlib_visualize.correlation_matrix")
def test_correlation_matrix(mock_draw):
    corr_mat = np.asarray([[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.1, 0.1, 1.0]])
    corr_mat_pruned = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    labels = ["a", "b", "c"]
    labels_pruned = ["a", "b"]
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "correlation_matrix.pdf"
    fit_results = fit.FitResults(np.empty(0), np.empty(0), labels, corr_mat, 1.0)

    # pruning with threshold
    visualize.correlation_matrix(
        fit_results, figure_folder=folder_path, pruning_threshold=0.15
    )

    mock_draw.assert_called_once()
    assert np.allclose(mock_draw.call_args[0][0], corr_mat_pruned)
    assert np.any(
        [mock_draw.call_args[0][1][i] == labels[i] for i in range(len(labels_pruned))]
    )
    assert mock_draw.call_args[0][2] == figure_path
    assert mock_draw.call_args[1] == {}

    # pruning of fixed parameter (all zeros in correlation matrix row/column)
    corr_mat_fixed = np.asarray([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]])
    fit_results_fixed = fit.FitResults(
        np.empty(0), np.empty(0), labels, corr_mat_fixed, 1.0
    )
    visualize.correlation_matrix(fit_results_fixed, figure_folder=folder_path)
    assert np.allclose(mock_draw.call_args_list[1][0][0], corr_mat_pruned)
    assert np.any(
        [
            mock_draw.call_args_list[1][0][1][i] == labels[i]
            for i in range(len(labels_pruned))
        ]
    )

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.correlation_matrix(
            fit_results, figure_folder=folder_path, method="unknown"
        )


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
        fit_results,
        figure_folder=folder_path,
        exclude_list=exclude_list,
        method="matplotlib",
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
    # and fixed parameter removal
    fit_results.uncertainty[0] = 0.0

    bestfit_expected = np.asarray([1.0, 1.1])
    uncertainty_expected = np.asarray([1.0, 0.7])
    labels_expected = ["b", "c"]
    visualize.pulls(fit_results, figure_folder=folder_path, method="matplotlib")

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
            fit_results,
            figure_folder=folder_path,
            exclude_list=exclude_list,
            method="unknown",
        )


@mock.patch("cabinetry.contrib.matplotlib_visualize.ranking")
def test_ranking(mock_draw):
    bestfit = np.asarray([1.2, 0.1])
    uncertainty = np.asarray([0.2, 0.8])
    labels = ["staterror_a", "modeling"]
    impact_prefit_up = np.asarray([0.1, 0.5])
    impact_prefit_down = np.asarray([-0.2, -0.4])
    impact_postfit_up = np.asarray([0.1, 0.4])
    impact_postfit_down = np.asarray([-0.2, -0.3])
    ranking_results = fit.RankingResults(
        bestfit,
        uncertainty,
        labels,
        impact_prefit_up,
        impact_prefit_down,
        impact_postfit_up,
        impact_postfit_down,
    )
    folder_path = "tmp"

    figure_path = pathlib.Path(folder_path) / "ranking.pdf"
    bestfit_expected = np.asarray([0.1, 1.2])
    uncertainty_expected = np.asarray([0.8, 0.2])
    labels_expected = ["modeling", "staterror_a"]

    visualize.ranking(ranking_results, figure_folder=folder_path)
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
    visualize.ranking(ranking_results, figure_folder=folder_path, max_pars=1)
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
        visualize.ranking(ranking_results, figure_folder=folder_path, method="unknown")


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
    nominal_path = tmp_path / "region_sample_Nominal_modified.npz"
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

    visualize.templates(config, figure_folder=folder_path)

    assert mock_histo_config.call_args_list == [
        [(tmp_path, region, sample, {"Name": "Nominal"}), {}]
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
        visualize.templates(config, figure_folder=folder_path, method="unknown")

    # remove files for variation histograms
    up_path.unlink()
    down_path.unlink()

    visualize.templates(config, figure_folder=folder_path)
    assert mock_draw.call_count == 1  # no new call, since no variations found


@mock.patch("cabinetry.contrib.matplotlib_visualize.scan")
def test_scan(mock_draw):
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "scan_a_0.pdf"

    par_name = "a[0]"
    par_mle = 1.5
    par_unc = 0.2
    par_vals = np.asarray([1.3, 1.5, 1.7])
    par_nlls = np.asarray([0.9, 0.0, 1.1])
    scan_results = fit.ScanResults(par_name, par_mle, par_unc, par_vals, par_nlls)

    visualize.scan(scan_results, figure_folder=folder_path)

    assert mock_draw.call_count == 1
    assert mock_draw.call_args[0][0] == par_name
    assert mock_draw.call_args[0][1] == par_mle
    assert mock_draw.call_args[0][2] == par_unc
    assert np.allclose(mock_draw.call_args[0][3], par_vals)
    assert np.allclose(mock_draw.call_args[0][4], par_nlls)
    assert mock_draw.call_args[0][5] == figure_path
    assert mock_draw.call_args[1] == {}

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.scan(scan_results, figure_folder=folder_path, method="unknown")


@mock.patch("cabinetry.contrib.matplotlib_visualize.limit")
def test_limit(mock_draw):
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "limit.pdf"

    observed_CLs = np.asarray([0.75, 0.32, 0.02])
    expected_CLs = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(3)])
    poi_values = np.asarray([0, 1, 2])
    limit_results = fit.LimitResults(
        3.0, np.empty(5), observed_CLs, expected_CLs, poi_values
    )

    visualize.limit(limit_results, figure_folder=folder_path)

    assert mock_draw.call_count == 1
    assert np.allclose(mock_draw.call_args[0][0], limit_results.observed_CLs)
    assert np.allclose(mock_draw.call_args[0][1], limit_results.expected_CLs)
    assert np.allclose(mock_draw.call_args[0][2], limit_results.poi_values)
    assert mock_draw.call_args[0][3] == figure_path
    assert mock_draw.call_args[1] == {}

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.limit(limit_results, figure_folder=folder_path, method="unknown")
