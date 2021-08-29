import copy
from unittest import mock

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import numpy as np

from cabinetry.visualize import plot_model


def test_data_mc(tmp_path):
    fname = tmp_path / "fig.pdf"
    histo_dict_list = [
        {
            "label": "Background",
            "isData": False,
            "yields": np.asarray([12.5, 14]),
            "variable": "x",
        },
        {
            "label": "Signal",
            "isData": False,
            "yields": np.asarray([2, 5]),
            "variable": "x",
        },
        {
            "label": "Data",
            "isData": True,
            "yields": np.asarray([13, 15]),
            "variable": "x",
        },
    ]
    total_model_unc = np.sqrt([0.17, 0.29])
    bin_edges = np.asarray([1, 2, 3])

    fig = plot_model.data_mc(
        histo_dict_list,
        total_model_unc,
        bin_edges,
        fname,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc.pdf", str(fname), 0) is None
    )

    # compare figure returned by function
    fname_ret = tmp_path / "fig_from_return.pdf"
    fig.savefig(fname_ret)
    assert (
        compare_images("tests/visualize/reference/data_mc.pdf", str(fname_ret), 0)
        is None
    )

    histo_dict_list_log = copy.deepcopy(histo_dict_list)
    histo_dict_list_log[0]["yields"] = np.asarray([2000, 14])
    histo_dict_list_log[2]["yields"] = np.asarray([2010, 15])
    total_model_unc_log = np.asarray([50, 1.5])
    bin_edges_log = np.asarray([10, 100, 1000])
    fname_log = fname.with_name(fname.stem + "_log" + fname.suffix)

    # automatic log scale
    plot_model.data_mc(
        histo_dict_list_log,
        total_model_unc_log,
        bin_edges_log,
        fname,
        log_scale_x=True,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc_log.pdf", str(fname_log), 0)
        is None
    )

    # linear scale forced
    plot_model.data_mc(
        histo_dict_list,
        total_model_unc,
        bin_edges,
        fname,
        log_scale=False,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc.pdf", str(fname), 0) is None
    )

    # log scale forced
    plot_model.data_mc(
        histo_dict_list_log,
        total_model_unc_log,
        bin_edges_log,
        fname,
        log_scale=True,
        log_scale_x=True,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc_log.pdf", str(fname_log), 0)
        is None
    )

    # do not save figure, but close it
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = fig = plot_model.data_mc(
            histo_dict_list_log,
            total_model_unc_log,
            bin_edges_log,
            label="",
            close_figure=True,
        )
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")


def test_templates(tmp_path):
    fname = tmp_path / "fig.pdf"
    nominal_histo = {
        "yields": np.asarray([1.0, 1.2]),
        "stdev": np.asarray([0.05, 0.06]),
    }
    up_histo_orig = {
        "yields": np.asarray([1.2, 1.7]),
        "stdev": np.asarray([0.05, 0.07]),
    }
    down_histo_orig = {
        "yields": np.asarray([0.9, 0.9]),
        "stdev": np.asarray([0.06, 0.07]),
    }
    up_histo_mod = {"yields": np.asarray([1.3, 1.6]), "stdev": np.asarray([0.05, 0.07])}
    down_histo_mod = {
        "yields": np.asarray([0.85, 0.95]),
        "stdev": np.asarray([0.06, 0.07]),
    }
    bin_edges = np.asarray([0.0, 1.0, 2.0])
    variable = "x"
    label = "region: Signal region\nsample: Signal\nsystematic: Modeling"

    fig = plot_model.templates(
        nominal_histo,
        up_histo_orig,
        down_histo_orig,
        up_histo_mod,
        down_histo_mod,
        bin_edges,
        variable,
        fname,
        label=label,
    )
    assert (
        compare_images("tests/visualize/reference/templates.pdf", str(fname), 0) is None
    )

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.pdf"
    fig.savefig(fname)
    assert (
        compare_images("tests/visualize/reference/templates.pdf", str(fname), 0) is None
    )

    # do not save figure, but close it
    # only single variation specified
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = plot_model.templates(
            nominal_histo,
            up_histo_orig,
            {},
            up_histo_mod,
            {},
            bin_edges,
            variable,
            label=label,
            close_figure=True,
        )
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    plt.close("all")
