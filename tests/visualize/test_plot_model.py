import copy

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import numpy as np

from cabinetry.visualize import plot_model


def test_no_open_figure():
    # ensure there are no open figures at the start, if this fails then some other part
    # of the test suite opened a figure without closing it
    assert len(plt.get_fignums()) == 0


def test_data_MC(tmp_path):
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
    plot_model.data_MC(
        histo_dict_list,
        total_model_unc,
        bin_edges,
        fname,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_MC.pdf", str(fname), 0) is None
    )

    histo_dict_list_log = copy.deepcopy(histo_dict_list)
    histo_dict_list_log[0]["yields"] = np.asarray([2000, 14])
    histo_dict_list_log[2]["yields"] = np.asarray([2010, 15])
    total_model_unc_log = np.asarray([50, 1.5])
    bin_edges_log = np.asarray([10, 100, 1000])
    fname_log = fname.with_name(fname.stem + "_log" + fname.suffix)

    # automatic log scale
    plot_model.data_MC(
        histo_dict_list_log,
        total_model_unc_log,
        bin_edges_log,
        fname,
        log_scale_x=True,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_MC_log.pdf", str(fname_log), 0)
        is None
    )

    # linear scale forced
    plot_model.data_MC(
        histo_dict_list,
        total_model_unc,
        bin_edges,
        fname,
        log_scale=False,
        label="Signal region\npre-fit",
    )
    assert (
        compare_images("tests/visualize/reference/data_MC.pdf", str(fname), 0) is None
    )

    # three open figures, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 3

    # log scale forced
    plot_model.data_MC(
        histo_dict_list_log,
        total_model_unc_log,
        bin_edges_log,
        fname,
        log_scale=True,
        log_scale_x=True,
        label="Signal region\npre-fit",
        close_figure=True,
    )
    assert (
        compare_images("tests/visualize/reference/data_MC_log.pdf", str(fname_log), 0)
        is None
    )
    assert len(plt.get_fignums()) == 3
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
    up_histo_mod = {
        "yields": np.asarray([1.3, 1.6]),
        "stdev": np.asarray([0.05, 0.07]),
    }
    down_histo_mod = {
        "yields": np.asarray([0.85, 0.95]),
        "stdev": np.asarray([0.06, 0.07]),
    }
    bin_edges = np.asarray([0.0, 1.0, 2.0])
    variable = "x"
    label = "region: Signal region\nsample: Signal\nsystematic: Modeling"

    plot_model.templates(
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

    # single open figure, does not change when calling with close_figure
    assert len(plt.get_fignums()) == 1

    # only single variation specified
    plot_model.templates(
        nominal_histo,
        up_histo_orig,
        {},
        up_histo_mod,
        {},
        bin_edges,
        variable,
        fname,
        label=label,
        close_figure=True,
    )
    assert len(plt.get_fignums()) == 1
    plt.close("all")
