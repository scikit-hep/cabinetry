import copy
from unittest import mock

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from cabinetry.visualize import plot_model


def test_data_mc(tmp_path, caplog):
    fname = tmp_path / "fig.png"
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
        figure_path=fname,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc.png", str(fname), 0) is None
    )

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert (
        compare_images("tests/visualize/reference/data_mc.png", str(fname), 0) is None
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
        figure_path=fname,
        log_scale_x=True,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc_log.png", str(fname_log), 0)
        is None
    )

    # linear scale forced
    plot_model.data_mc(
        histo_dict_list,
        total_model_unc,
        bin_edges,
        figure_path=fname,
        log_scale=False,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc.png", str(fname), 0) is None
    )

    # log scale forced
    plot_model.data_mc(
        histo_dict_list_log,
        total_model_unc_log,
        bin_edges_log,
        figure_path=fname,
        log_scale=True,
        log_scale_x=True,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_mc_log.png", str(fname_log), 0)
        is None
    )

    # do not save figure, but close it
    # one bin with zero model prediction
    histo_dict_list[0]["yields"] = np.asarray([0, 14])
    histo_dict_list[1]["yields"] = np.asarray([0, 5])
    caplog.clear()
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        with pytest.warns() as warn_record:
            fig = fig = plot_model.data_mc(
                histo_dict_list,
                total_model_unc_log,
                bin_edges_log,
                label="",
                close_figure=True,
            )
            assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    assert "predicted yield is zero in 1 bin(s), excluded from ratio plot" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # expect three RuntimeWarnings from numpy due to division by zero
    assert sum("divide by zero" in str(m.message) for m in warn_record) == 3

    plt.close("all")


def test_templates(tmp_path):
    fname = tmp_path / "fig.png"
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
        figure_path=fname,
        label=label,
    )
    assert (
        compare_images("tests/visualize/reference/templates.png", str(fname), 0) is None
    )

    # compare figure returned by function
    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert (
        compare_images("tests/visualize/reference/templates.png", str(fname), 0) is None
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


def test_modifier_grid(tmp_path):
    fname = tmp_path / "fig.png"
    grid_list = [
        np.asarray([[5, 4, 8, 2, 1, 3, 8], [7, 4, 8, 8, 8, 3, 0]]),
        np.asarray([[5, 8, 4, 2, 1, 3, 8], [6, 8, 4, 8, 8, 3, 0]]),
    ]
    axis_labels = [
        ["SR", "CR"],
        ["Background", "Signal"],
        ["a", "b", "c", "d", "e", "f", "mu"],
    ]
    category_map = {
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

    fig = plot_model.modifier_grid(
        grid_list, axis_labels, category_map, figure_path=fname
    )
    # non-zero tolerance as layout changes very slightly in CI
    assert (
        compare_images("tests/visualize/reference/modifier_grid.png", str(fname), 16)
        is None
    )

    # saving the return figure again seems to slightly change formatting, so instead of
    # doing that here in the test (like in the tests above), do not save the figure
    # automatically and only manually save it in the next test to compare

    # do not save figure, but close it and save returned figure manually
    with mock.patch("cabinetry.visualize.utils._save_and_close") as mock_close_safe:
        fig = plot_model.modifier_grid(
            grid_list, axis_labels, category_map, close_figure=True
        )
        assert mock_close_safe.call_args_list == [((fig, None, True), {})]

    fname = tmp_path / "fig_from_return.png"
    fig.savefig(fname)
    assert (
        compare_images("tests/visualize/reference/modifier_grid.png", str(fname), 16)
        is None
    )

    plt.close("all")
