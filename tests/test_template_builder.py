from pathlib import Path

import logging
import numpy as np
import pytest

from cabinetry import histo
from cabinetry import template_builder


def test__get_ntuple_path():
    assert template_builder._get_ntuple_path(
        {"Path": "test/path.root"}, {}, {"Name": "nominal"}
    ) == Path("test/path.root")

    assert template_builder._get_ntuple_path(
        {},
        {},
        {"Name": "variation", "Type": "NormPlusShape", "PathUp": "test/path.root"},
    ) == Path("test/path.root")

    with pytest.raises(
        NotImplementedError, match="ntuple path treatment not yet defined"
    ) as e_info:
        assert template_builder._get_ntuple_path(
            {}, {}, {"Name": "unknown_variation_type", "Type": "unknown"}
        )


def test__get_variable():
    assert template_builder._get_variable({}, {"Variable": "jet_pt"}, {}) == "jet_pt"


def test__get_filter():
    assert (
        template_builder._get_filter({}, {"Filter": "jet_pt > 0"}, {}) == "jet_pt > 0"
    )
    assert template_builder._get_filter({}, {}, {}) is None


def test__get_weight():
    assert template_builder._get_weight({"Weight": "weight_mc"}, {}, {}) == "weight_mc"
    assert template_builder._get_weight({}, {}, {}) is None


def test__get_position_in_file():
    assert (
        template_builder._get_position_in_file(
            {"Tree": "tree_name"}, {"Name": "nominal"}
        )
        == "tree_name"
    )

    assert (
        template_builder._get_position_in_file(
            {}, {"Name": "variation", "Type": "NormPlusShape", "TreeUp": "up_tree"}
        )
        == "up_tree"
    )

    with pytest.raises(
        NotImplementedError, match="ntuple path treatment not yet defined"
    ) as e_info:
        template_builder._get_position_in_file(
            {}, {"Name": "unknown_variation", "Type": "unknown"}
        )


def test__get_binning():
    np.testing.assert_equal(
        template_builder._get_binning({"Binning": [1, 2, 3]}), [1, 2, 3]
    )
    with pytest.raises(NotImplementedError, match="cannot determine binning") as e_info:
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

    config = {
        "Samples": [{"Name": "sample", "Tree": treename, "Path": fname}],
        "Regions": [{"Name": "test_region", "Variable": varname, "Binning": bins}],
        "Systematics": [],
    }
    template_builder.create_histograms(config, tmp_path, method="uproot")

    # ServiceX is not yet implemented
    with pytest.raises(
        NotImplementedError, match="ServiceX not yet implemented"
    ) as e_info:
        assert template_builder.create_histograms(config, tmp_path, method="ServiceX")

    # other backends
    with pytest.raises(NotImplementedError, match="unknown backend") as e_info:
        assert template_builder.create_histograms(config, tmp_path, method="unknown")

    saved_histo, _ = histo.load_from_config(
        tmp_path,
        config["Samples"][0],
        config["Regions"][0],
        {"Name": "nominal"},
        modified=False,
    )

    assert np.allclose(saved_histo["yields"], [1, 1, 2])
    assert np.allclose(saved_histo["sumw2"], [1, 1, 1.41421356])
    assert np.allclose(saved_histo["bins"], bins)

    assert "  reading sample sample" in [rec.message for rec in caplog.records]
    assert "  in region test_region" in [rec.message for rec in caplog.records]
    assert "  variation nominal" in [rec.message for rec in caplog.records]
    caplog.clear()
