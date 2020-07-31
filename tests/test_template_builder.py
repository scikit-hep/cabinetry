import logging
import pathlib

import numpy as np
import pytest

from cabinetry import histo
from cabinetry import template_builder


def test__check_for_override():
    # override exists for template
    assert (
        template_builder._check_for_override(
            {"Up": {"setting": "val"}}, "Up", "setting"
        )
        == "val"
    )

    # no override for template
    assert template_builder._check_for_override({}, "Up", "setting") is None

    # no option requested
    assert (
        template_builder._check_for_override({"Up": {"setting": "val"}}, "Up", "")
        is None
    )

    # no template requested
    assert (
        template_builder._check_for_override({"Up": {"setting": "val"}}, "", "setting")
        is None
    )


def test__get_ntuple_path():
    # no override
    assert template_builder._get_ntuple_path(
        {}, {"Path": "path.root"}, {"Name": "nominal"}, ""
    ) == pathlib.Path("path.root")

    # systematic with override
    assert template_builder._get_ntuple_path(
        {},
        {"Path": "path.root"},
        {"Name": "variation", "Up": {"Path": "variation.root"}},
        "Up",
    ) == pathlib.Path("variation.root")

    # systematic without override
    assert template_builder._get_ntuple_path(
        {}, {"Path": "path.root"}, {"Name": "variation"}, "Up"
    ) == pathlib.Path("path.root")


def test__get_variable():
    assert template_builder._get_variable({"Variable": "jet_pt"}) == "jet_pt"


def test__get_filter():
    # no override
    assert (
        template_builder._get_filter({"Filter": "jet_pt > 0"}, {}, {}, "")
        == "jet_pt > 0"
    )

    # no filter
    assert template_builder._get_filter({}, {}, {}, "") is None

    # systematic with override
    assert (
        template_builder._get_filter(
            {"Filter": "jet_pt > 0"},
            {},
            {"Name": "variation", "Up": {"Filter": "jet_pt > 100"}},
            "Up",
        )
        == "jet_pt > 100"
    )

    # systematic without override
    assert (
        template_builder._get_filter(
            {"Filter": "jet_pt > 0"}, {}, {"Name": "variation"}, "Up",
        )
        == "jet_pt > 0"
    )


def test__get_weight():
    # no override
    assert (
        template_builder._get_weight({}, {"Weight": "weight_mc"}, {}, "") == "weight_mc"
    )

    # no weight
    assert template_builder._get_weight({}, {}, {}, "") is None

    # systematic with override
    assert (
        template_builder._get_weight(
            {},
            {"Weight": "weight_mc"},
            {"Name": "variation", "Up": {"Weight": "weight_modified"}},
            "Up",
        )
        == "weight_modified"
    )

    # systematic without override
    assert (
        template_builder._get_weight(
            {}, {"Weight": "weight_mc"}, {"Name": "variation"}, "Up",
        )
        == "weight_mc"
    )


def test__get_position_in_file():
    # no override
    assert (
        template_builder._get_position_in_file(
            {"Tree": "tree_name"}, {"Name": "nominal"}, ""
        )
        == "tree_name"
    )

    # systematic with override
    assert (
        template_builder._get_position_in_file(
            {"Tree": "nominal"}, {"Name": "variation", "Up": {"Tree": "up_tree"}}, "Up",
        )
        == "up_tree"
    )

    # systematic without override
    assert (
        template_builder._get_position_in_file(
            {"Tree": "nominal"}, {"Name": "variation"}, "Up",
        )
        == "nominal"
    )


def test__get_binning():
    np.testing.assert_equal(
        template_builder._get_binning({"Binning": [1, 2, 3]}), [1, 2, 3]
    )
    with pytest.raises(NotImplementedError, match="cannot determine binning"):
        assert template_builder._get_binning({})


def test_create_histograms(tmp_path, caplog, utils):
    caplog.set_level(logging.DEBUG)
    fname = tmp_path / "test.root"
    treename = "tree"
    varname = "var"
    var_array = [1.1, 2.3, 3.0, 3.2]
    weightname = "weight"
    weight_array = [1.0, 1.0, 2.0, 1.0]
    bins = [1, 2, 3, 4]
    # create something to read
    utils.create_ntuple(fname, treename, varname, var_array, weightname, weight_array)

    # create a systematic uncertainty to read
    var_array_sys = [1.1, 1.1, 1.1, 1.1]
    fname_sys = tmp_path / "test_sys.root"
    utils.create_ntuple(
        fname_sys, treename, varname, var_array_sys, weightname, weight_array
    )

    config = {
        "Regions": [{"Name": "test_region", "Variable": varname, "Binning": bins}],
        "Samples": [{"Name": "sample", "Tree": treename, "Path": fname}],
        "Systematics": [
            {"Name": "norm", "Type": "Normalization"},
            {
                "Name": "var",
                "Type": "NormPlusShape",
                "Samples": "sample",
                "Up": {"Path": str(fname_sys)},
            },
        ],
    }
    template_builder.create_histograms(config, tmp_path, method="uproot")

    # ServiceX is not yet implemented
    with pytest.raises(NotImplementedError, match="ServiceX not yet implemented"):
        assert template_builder.create_histograms(config, tmp_path, method="ServiceX")

    # other backends
    with pytest.raises(NotImplementedError, match="unknown backend"):
        assert template_builder.create_histograms(config, tmp_path, method="unknown")

    saved_histo = histo.Histogram.from_config(
        tmp_path,
        config["Regions"][0],
        config["Samples"][0],
        {"Name": "nominal"},
        modified=False,
    )

    assert np.allclose(saved_histo.yields, [1, 1, 2])
    assert np.allclose(saved_histo.stdev, [1, 1, 1.41421356])
    assert np.allclose(saved_histo.bins, bins)

    saved_histo_sys = histo.Histogram.from_config(
        tmp_path,
        config["Regions"][0],
        config["Samples"][0],
        {"Name": "var"},
        modified=False,
        template="Up",
    )

    assert np.allclose(saved_histo_sys.yields, [4, 0, 0])
    assert np.allclose(saved_histo_sys.stdev, [2, 0, 0])
    assert np.allclose(saved_histo_sys.bins, bins)

    assert "  reading sample sample" in [rec.message for rec in caplog.records]
    assert "  in region test_region" in [rec.message for rec in caplog.records]
    assert "  variation nominal" in [rec.message for rec in caplog.records]
    assert "  variation var" in [rec.message for rec in caplog.records]
    caplog.clear()
