from collections import defaultdict, namedtuple
import copy
import logging
import pathlib
from unittest import mock

import matplotlib.figure
import numpy as np
import pyhf
import pytest

from cabinetry import fit
from cabinetry import model_utils
from cabinetry import visualize


MockHistogram = namedtuple("MockHistogram", ["bins", "yields", "stdev"])


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("SR", "pre-fit"), "SR_prefit.pdf"),
        (("SR", "post-fit"), "SR_postfit.pdf"),
        (("SR 1", "prefit"), "SR-1_prefit.pdf"),
        (("SR 1", "postfit"), "SR-1_postfit.pdf"),
    ],
)
def test__figure_name(test_input, expected):
    assert visualize._figure_name(*test_input) == expected


def test__total_yield_uncertainty():
    stdev_list = [np.asarray([0.1, 0.2, 0.1]), np.asarray([0.3, 0.2, 0.1])]
    expected_uncertainties = [0.31622777, 0.28284271, 0.14142136]
    assert np.allclose(
        visualize._total_yield_uncertainty(stdev_list), expected_uncertainties
    )


@mock.patch("cabinetry.visualize._total_yield_uncertainty", return_value=[0.2])
@mock.patch(
    "cabinetry.visualize.plot_model.data_mc", return_value=matplotlib.figure.Figure()
)
@mock.patch(
    "cabinetry.histo.Histogram.from_config",
    return_value=MockHistogram([0.0, 1.0], [1.0], [0.1]),
)
def test_data_mc_from_histograms(mock_load, mock_draw, mock_stdev):
    config = {
        "General": {"HistogramFolder": "tmp_hist"},
        "Regions": [{"Name": "reg_1", "Variable": "x"}],
        "Samples": [{"Name": "sample_1"}, {"Name": "data", "Data": True}],
    }
    figure_folder = pathlib.Path("tmp")
    histogram_folder = pathlib.Path("tmp_hist")

    fig_dict_list = visualize.data_mc_from_histograms(
        config, figure_folder=figure_folder
    )
    assert len(fig_dict_list) == 1
    assert isinstance(fig_dict_list[0]["figure"], matplotlib.figure.Figure)
    assert fig_dict_list[0]["region"] == "reg_1"

    # the call_args_list contains calls (outer round brackets), first filled with
    # arguments (inner round brackets) and then keyword arguments
    assert mock_load.call_args_list == [
        (
            (
                histogram_folder,
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "data", "Data": True},
                {},
            ),
            {"modified": True},
        ),
        (
            (
                histogram_folder,
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "sample_1"},
                {},
            ),
            {"modified": True},
        ),
    ]
    assert mock_stdev.call_args_list == [(([[0.1]],), {})]
    assert mock_draw.call_args_list == [
        (
            (
                [
                    {"label": "data", "isData": True, "yields": [1.0], "variable": "x"},
                    {
                        "label": "sample_1",
                        "isData": False,
                        "yields": [1.0],
                        "variable": "x",
                    },
                ],
                [0.2],
                [0.0, 1.0],
            ),
            {
                "figure_path": figure_folder / "reg_1_prefit.pdf",
                "log_scale": None,
                "log_scale_x": False,
                "label": "reg_1\npre-fit",
                "close_figure": False,
            },
        )
    ]

    # custom log scale settings, close figure, do not save figure
    _ = visualize.data_mc_from_histograms(
        config,
        figure_folder=figure_folder,
        log_scale=True,
        log_scale_x=True,
        close_figure=True,
        save_figure=False,
    )
    assert mock_draw.call_args[1] == {
        "figure_path": None,
        "log_scale": True,
        "log_scale_x": True,
        "label": "reg_1\npre-fit",
        "close_figure": True,
    }


@mock.patch(
    "cabinetry.visualize.plot_model.data_mc", return_value=matplotlib.figure.Figure()
)
@mock.patch("cabinetry.templates.builder._binning", return_value=np.asarray([1, 2]))
@mock.patch(
    "cabinetry.configuration.region_dict",
    side_effect=[
        {"Name": "region", "Variable": "x", "Binning": [1, 2]},
        {"Name": "region"},
    ],
)
@mock.patch(
    "cabinetry.model_utils._filter_channels",
    side_effect=[["Signal Region"], ["Signal Region"], ["Signal Region"], []],
)
@mock.patch("cabinetry.model_utils._data_per_channel", return_value=[[12.0]])
def test_data_mc(mock_data, mock_filter, mock_dict, mock_bins, mock_draw, example_spec):
    config = {"config": "abc"}
    figure_folder = "tmp"
    model = pyhf.Workspace(example_spec).model()
    model_pred = model_utils.ModelPrediction(
        model, [[[10.0]]], [[[0.3], [0.3]]], [[0.3, 0.3]], "pre-fit"
    )
    data = [12.0, 1.0]

    # pre-fit plot
    fig_dict_list = visualize.data_mc(
        model_pred, data, config=config, figure_folder=figure_folder
    )
    assert len(fig_dict_list) == 1
    assert isinstance(fig_dict_list[0]["figure"], matplotlib.figure.Figure)
    assert fig_dict_list[0]["region"] == "Signal Region"

    assert mock_data.call_args_list == [((model, data), {})]
    assert mock_filter.call_args_list == [((model, None), {})]
    assert mock_dict.call_args_list == [((config, "Signal Region"), {})]
    assert mock_bins.call_args_list == [
        (({"Name": "region", "Variable": "x", "Binning": [1, 2]},), {})
    ]

    expected_histograms = [
        {
            "label": "Signal",
            "isData": False,
            "yields": np.asarray([10.0]),
            "variable": "x",
        },
        {
            "label": "Data",
            "isData": True,
            "yields": np.asarray([12.0]),
            "variable": "x",
        },
    ]
    assert mock_draw.call_count == 1
    assert mock_draw.call_args_list[0][0][0] == expected_histograms
    assert np.allclose(mock_draw.call_args_list[0][0][1], np.asarray([0.3]))
    np.testing.assert_equal(mock_draw.call_args_list[0][0][2], np.asarray([1, 2]))
    assert mock_draw.call_args_list[0][1] == {
        "figure_path": pathlib.Path("tmp/Signal-Region_prefit.pdf"),
        "log_scale": None,
        "log_scale_x": False,
        "label": "Signal Region\npre-fit",
        "close_figure": False,
    }

    # post-fit plot (different label in model prediction), custom scale, close figure,
    # do not save figure, histogram input mode: no binning or variable specified (via
    # side effect)
    model_pred = model_utils.ModelPrediction(
        model, [[[11.0]]], [[[0.2], [0.2]]], [[0.2, 0.2]], "post-fit"
    )
    _ = visualize.data_mc(
        model_pred,
        data,
        config=config,
        figure_folder=figure_folder,
        log_scale=False,
        close_figure=True,
        save_figure=False,
    )

    assert mock_draw.call_count == 2
    # yield at best-fit point is different from pre-fit
    assert np.allclose(mock_draw.call_args[0][0][0]["yields"], 11.0)
    assert np.allclose(mock_draw.call_args[0][1], np.asarray([0.2]))
    # observable defaults to "bin"
    assert mock_draw.call_args[0][0][0]["variable"] == "bin"
    assert mock_draw.call_args[0][0][1]["variable"] == "bin"
    # binning falls back to default
    np.testing.assert_equal(mock_draw.call_args[0][2], np.asarray([0, 1]))
    assert mock_draw.call_args[1] == {
        "figure_path": None,  # figure not saved
        "log_scale": False,
        "log_scale_x": False,
        "label": "Signal Region\npost-fit",
        "close_figure": True,
    }

    # no config specified, default variable name and bin edges
    _ = visualize.data_mc(model_pred, data)
    assert mock_draw.call_args[0][0][0]["variable"] == "bin"
    assert mock_draw.call_args[0][0][1]["variable"] == "bin"
    assert mock_draw.call_args[0][0][1]["yields"] == np.asarray(data[:1])
    np.testing.assert_equal(mock_draw.call_args[0][2], np.asarray([0, 1]))

    # no matching channels (via side_effect)
    assert visualize.data_mc(model_pred, data, channels="abc") is None
    assert mock_filter.call_args == ((model, "abc"), {})
    assert mock_draw.call_count == 3  # no new call


@mock.patch(
    "cabinetry.histo.Histogram.from_path",
    side_effect=[
        MockHistogram([0.0, 1.0], [2.0], [0.2]),
        MockHistogram([0.0, 1.0], [3.0], [0.3]),
        MockHistogram([0.0, 1.0], [4.0], [0.4]),
        MockHistogram([0.0, 1.0], [5.0], [0.5]),
    ]
    * 3,
)
@mock.patch(
    "cabinetry.histo.Histogram.from_config",
    return_value=MockHistogram([0.0, 1.0], [1.0], [0.1]),
)
@mock.patch(
    "cabinetry.visualize.plot_model.templates", return_value=matplotlib.figure.Figure()
)
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

    fig_dict_list = visualize.templates(config, figure_folder=folder_path)
    assert len(fig_dict_list) == 1
    assert isinstance(fig_dict_list[0]["figure"], matplotlib.figure.Figure)
    assert fig_dict_list[0]["region"] == "region"
    assert fig_dict_list[0]["sample"] == "sample"
    assert fig_dict_list[0]["systematic"] == "sys"

    # nominal histogram loading
    assert mock_histo_config.call_args_list == [((tmp_path, region, sample, {}), {})]
    # variation histograms
    down_path_orig = pathlib.Path(str(down_path).replace("_modified", ""))
    up_path_orig = pathlib.Path(str(up_path).replace("_modified", ""))
    assert mock_histo_path.call_args_list == [
        ((down_path_orig,), {}),
        ((down_path,), {}),
        ((up_path_orig,), {}),
        ((up_path,), {}),
    ]

    nominal = {"yields": [1.0], "stdev": [0.1]}
    up_orig = {"yields": [4.0], "stdev": [0.4]}
    down_orig = {"yields": [2.0], "stdev": [0.2]}
    up_mod = {"yields": [5.0], "stdev": [0.5]}
    down_mod = {"yields": [3.0], "stdev": [0.3]}
    bins = [0.0, 1.0]
    assert mock_draw.call_args_list == [
        (
            (nominal, up_orig, down_orig, up_mod, down_mod, bins, "x"),
            {
                "figure_path": figure_path,
                "label": "region: region\nsample: sample\nsystematic: sys",
                "close_figure": False,
            },
        )
    ]

    # close figure, do not save figure, and remove variable information from config
    # (simulating histogram inputs), so variable defaults to "observable"
    histo_config = copy.deepcopy(config)
    histo_config["Regions"] = [{"Name": "region"}]
    _ = visualize.templates(
        histo_config, figure_folder=folder_path, close_figure=True, save_figure=False
    )
    assert mock_draw.call_args == (
        (nominal, up_orig, down_orig, up_mod, down_mod, bins, "observable"),
        {
            "figure_path": None,
            "label": "region: region\nsample: sample\nsystematic: sys",
            "close_figure": True,
        },
    )

    # remove files for variation histograms
    up_path.unlink()
    down_path.unlink()

    assert mock_draw.call_count == 2  # two calls so far
    _ = visualize.templates(config, figure_folder=folder_path)
    assert mock_draw.call_count == 2  # no new call, since no variations found

    # no systematics in config
    config = {
        "General": {"HistogramFolder": tmp_path},
        "Regions": [region],
        "Samples": [sample, {"Name": "data", "Data": True}],
    }
    _ = visualize.templates(config, figure_folder=folder_path)
    assert mock_draw.call_count == 2  # no systematics, so no new calls


@mock.patch(
    "cabinetry.visualize.plot_result.correlation_matrix",
    return_value=matplotlib.figure.Figure(),
)
def test_correlation_matrix(mock_draw):
    corr_mat = np.asarray([[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.1, 0.1, 1.0]])
    corr_mat_pruned = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    labels = ["a", "b", "c"]
    labels_pruned = ["a", "b"]
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "correlation_matrix.pdf"
    fit_results = fit.FitResults(np.empty(0), np.empty(0), labels, corr_mat, 1.0)

    # pruning with threshold
    fig = visualize.correlation_matrix(
        fit_results, figure_folder=folder_path, pruning_threshold=0.15
    )
    assert isinstance(fig, matplotlib.figure.Figure)

    mock_draw.assert_called_once()
    assert np.allclose(mock_draw.call_args[0][0], corr_mat_pruned)
    assert np.any(
        [mock_draw.call_args[0][1][i] == labels[i] for i in range(len(labels_pruned))]
    )
    assert mock_draw.call_args[1] == {"figure_path": figure_path, "close_figure": True}

    # pruning of fixed parameter (all zeros in correlation matrix row/column), do not
    # close figure, do not save
    corr_mat_fixed = np.asarray([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]])
    fit_results_fixed = fit.FitResults(
        np.empty(0), np.empty(0), labels, corr_mat_fixed, 1.0
    )
    _ = visualize.correlation_matrix(
        fit_results_fixed,
        figure_folder=folder_path,
        close_figure=False,
        save_figure=False,
    )
    assert np.allclose(mock_draw.call_args_list[1][0][0], corr_mat_pruned)
    assert np.any(
        [
            mock_draw.call_args_list[1][0][1][i] == labels[i]
            for i in range(len(labels_pruned))
        ]
    )
    assert mock_draw.call_args[1] == {"figure_path": None, "close_figure": False}


@mock.patch(
    "cabinetry.visualize.plot_result.pulls", return_value=matplotlib.figure.Figure()
)
def test_pulls(mock_draw):
    bestfit = np.asarray([0.8, 1.0, 1.05, 1.1])
    uncertainty = np.asarray([0.9, 1.0, 0.03, 0.7])
    labels = ["a", "b", "staterror_region[0]", "c"]
    exclude = ["a"]
    folder_path = "tmp"
    fit_results = fit.FitResults(bestfit, uncertainty, labels, np.empty(0), 1.0)

    filtered_bestfit = np.asarray([1.0, 1.1])
    filtered_uncertainty = np.asarray([1.0, 0.7])
    filtered_labels = np.asarray(["b", "c"])
    figure_path = pathlib.Path(folder_path) / "pulls.pdf"

    # with filtering
    fig = visualize.pulls(fit_results, figure_folder=folder_path, exclude=exclude)
    assert isinstance(fig, matplotlib.figure.Figure)

    mock_draw.assert_called_once()
    assert np.allclose(mock_draw.call_args[0][0], filtered_bestfit)
    assert np.allclose(mock_draw.call_args[0][1], filtered_uncertainty)
    assert np.any(
        [
            mock_draw.call_args[0][2][i] == filtered_labels[i]
            for i in range(len(filtered_labels))
        ]
    )
    assert mock_draw.call_args[1] == {"figure_path": figure_path, "close_figure": True}

    # filtering single parameter instead of list
    _ = visualize.pulls(fit_results, figure_folder=folder_path, exclude=exclude[0])

    assert np.allclose(mock_draw.call_args[0][0], filtered_bestfit)
    assert np.allclose(mock_draw.call_args[0][1], filtered_uncertainty)
    assert np.any(
        [
            mock_draw.call_args[0][2][i] == filtered_labels[i]
            for i in range(len(filtered_labels))
        ]
    )

    # without filtering via list, but with staterror removal, fixed parameter removal,
    # not closing figure, not saving
    fit_results.uncertainty[0] = 0.0

    bestfit_expected = np.asarray([1.0, 1.1])
    uncertainty_expected = np.asarray([1.0, 0.7])
    labels_expected = ["b", "c"]
    visualize.pulls(
        fit_results, figure_folder=folder_path, close_figure=False, save_figure=False
    )

    assert np.allclose(mock_draw.call_args[0][0], bestfit_expected)
    assert np.allclose(mock_draw.call_args[0][1], uncertainty_expected)
    assert np.any(
        [
            mock_draw.call_args[0][2][i] == labels_expected[i]
            for i in range(len(labels_expected))
        ]
    )
    assert mock_draw.call_args[1] == {"figure_path": None, "close_figure": False}


@mock.patch(
    "cabinetry.visualize.plot_result.ranking", return_value=matplotlib.figure.Figure()
)
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

    fig = visualize.ranking(ranking_results, figure_folder=folder_path)
    assert isinstance(fig, matplotlib.figure.Figure)

    assert mock_draw.call_count == 1
    assert np.allclose(mock_draw.call_args[0][0], bestfit_expected)
    assert np.allclose(mock_draw.call_args[0][1], uncertainty_expected)
    for i_lab, label in enumerate(mock_draw.call_args[0][2]):
        assert label == labels_expected[i_lab]
    assert np.allclose(mock_draw.call_args[0][3], impact_prefit_up[::-1])
    assert np.allclose(mock_draw.call_args[0][4], impact_prefit_down[::-1])
    assert np.allclose(mock_draw.call_args[0][5], impact_postfit_up[::-1])
    assert np.allclose(mock_draw.call_args[0][6], impact_postfit_down[::-1])
    assert mock_draw.call_args[1] == {"figure_path": figure_path, "close_figure": True}

    # maximum parameter amount specified, do not close figure, do not save figure
    _ = visualize.ranking(
        ranking_results,
        figure_folder=folder_path,
        max_pars=1,
        close_figure=False,
        save_figure=False,
    )
    assert mock_draw.call_count == 2
    assert np.allclose(mock_draw.call_args[0][0], bestfit_expected[0])
    assert np.allclose(mock_draw.call_args[0][1], uncertainty_expected[0])
    assert mock_draw.call_args[0][2] == labels_expected[0]
    assert np.allclose(mock_draw.call_args[0][3], impact_prefit_up[1])
    assert np.allclose(mock_draw.call_args[0][4], impact_prefit_down[1])
    assert np.allclose(mock_draw.call_args[0][5], impact_postfit_up[1])
    assert np.allclose(mock_draw.call_args[0][6], impact_postfit_down[1])
    assert mock_draw.call_args[1] == {"figure_path": None, "close_figure": False}


@mock.patch(
    "cabinetry.visualize.plot_result.scan", return_value=matplotlib.figure.Figure()
)
def test_scan(mock_draw):
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "scan_a_0.pdf"

    par_name = "a[0]"
    par_mle = 1.5
    par_unc = 0.2
    par_vals = np.asarray([1.3, 1.5, 1.7])
    par_nlls = np.asarray([0.9, 0.0, 1.1])
    scan_results = fit.ScanResults(par_name, par_mle, par_unc, par_vals, par_nlls)

    fig = visualize.scan(scan_results, figure_folder=folder_path)
    assert isinstance(fig, matplotlib.figure.Figure)

    assert mock_draw.call_count == 1
    assert mock_draw.call_args[0][0] == par_name
    assert mock_draw.call_args[0][1] == par_mle
    assert mock_draw.call_args[0][2] == par_unc
    assert np.allclose(mock_draw.call_args[0][3], par_vals)
    assert np.allclose(mock_draw.call_args[0][4], par_nlls)
    assert mock_draw.call_args[1] == {"figure_path": figure_path, "close_figure": True}

    # do not close figure, do not save figure
    _ = visualize.scan(
        scan_results, figure_folder=folder_path, close_figure=False, save_figure=False
    )
    assert mock_draw.call_args[1] == {"figure_path": None, "close_figure": False}


@mock.patch(
    "cabinetry.visualize.plot_result.limit", return_value=matplotlib.figure.Figure()
)
def test_limit(mock_draw):
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "limit.pdf"

    observed_CLs = np.asarray([0.75, 0.32, 0.02])
    expected_CLs = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(3)])
    poi_values = np.asarray([0, 1, 2])
    confidence_level = 0.90
    limit_results = fit.LimitResults(
        3.0, np.empty(5), observed_CLs, expected_CLs, poi_values, confidence_level
    )

    fig = visualize.limit(limit_results, figure_folder=folder_path)
    assert isinstance(fig, matplotlib.figure.Figure)

    assert mock_draw.call_count == 1
    assert np.allclose(mock_draw.call_args[0][0], limit_results.observed_CLs)
    assert np.allclose(mock_draw.call_args[0][1], limit_results.expected_CLs)
    assert np.allclose(mock_draw.call_args[0][2], limit_results.poi_values)
    assert np.allclose(mock_draw.call_args[0][3], 1 - limit_results.confidence_level)
    assert mock_draw.call_args[1] == {"figure_path": figure_path, "close_figure": True}

    # do not close figure, do not save figure
    _ = visualize.limit(
        limit_results, figure_folder=folder_path, close_figure=False, save_figure=False
    )
    assert mock_draw.call_args[1] == {"figure_path": None, "close_figure": False}


@mock.patch(
    "cabinetry.visualize.plot_model.modifier_grid",
    return_value=matplotlib.figure.Figure(),
)
@mock.patch(
    "cabinetry.model_utils._modifier_map",
    return_value=defaultdict(
        list,
        {("Signal Region", "Signal", "Signal strength"): ["normfactor"]},
    ),
)
def test_modifier_grid(mock_map, mock_draw, example_spec, caplog):
    caplog.set_level(logging.DEBUG)
    folder_path = "tmp"
    figure_path = pathlib.Path(folder_path) / "modifier_grid.pdf"

    model = pyhf.Workspace(example_spec).model()
    # model contains a staterror, but remove that from the map via mocked _modifier_map
    # return to capture the effect of fields in the grid without modifiers

    fig = visualize.modifier_grid(model, figure_folder=folder_path)
    assert isinstance(fig, matplotlib.figure.Figure)

    assert mock_map.call_count == 1
    assert mock_map.call_args[0][0].spec == model.spec
    assert mock_map.call_args[1] == {}

    assert mock_draw.call_count == 1
    assert np.allclose(mock_draw.call_args[0][0], [np.asarray([[0.0, 8.0]])])
    assert mock_draw.call_args[0][1] == [
        ["Signal Region"],
        ["Signal"],
        ["Signal strength", "staterror_Signal-Region"],
    ]
    assert mock_draw.call_args[0][2] == {
        0: "normfactor",
        1: "shapefactor",
        2: "shapesys",
        3: "lumi",
        4: "staterror",
        5: "normsys + histosys",
        6: "histosys",
        7: "normsys",
        8: "none",
    }
    assert mock_draw.call_args[1] == {"figure_path": figure_path, "close_figure": True}

    # do not close figure, do not save figure, split by sample
    _ = visualize.modifier_grid(
        model,
        figure_folder=folder_path,
        split_by_sample=True,
        close_figure=False,
        save_figure=False,
    )
    assert np.allclose(mock_draw.call_args[0][0], [np.asarray([[0.0, 8.0]])])
    assert mock_draw.call_args[1] == {"figure_path": None, "close_figure": False}
    assert mock_draw.call_args[0][1] == [
        ["Signal"],
        ["Signal Region"],
        ["Signal strength", "staterror_Signal-Region"],
    ]  # order changed due to split_by_sample
    caplog.clear()

    # unknown modifier combination: patch modifier map to return normfactor+staterror
    model = mock.MagicMock()
    model.config.channels = ["Signal Region"]
    model.config.samples = ["Signal"]
    model.config.par_order = ["Signal strength"]
    with mock.patch(
        "cabinetry.model_utils._modifier_map",
        return_value=defaultdict(
            list,
            {
                ("Signal Region", "Signal", "Signal strength"): [
                    "normfactor",
                    "staterror",
                ]
            },
        ),
    ):
        with pytest.raises(KeyError, match=r"staterror \+ normfactor"):
            _ = visualize.modifier_grid(model)

    assert "modifiers for Signal Region, Signal, Signal strength not supported" not in [
        rec.message for rec in caplog.records
    ]
