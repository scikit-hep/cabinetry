from pathlib import Path
import pytest

import numpy as np

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

    with pytest.raises(Exception) as e_info:
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

    with pytest.raises(Exception) as e_info:
        template_builder._get_position_in_file(
            {}, {"Name": "unknown_variation", "Type": "unknown"}
        )


def test__get_binning():
    np.testing.assert_equal(
        template_builder._get_binning({"Binning": [1, 2, 3]}), [1, 2, 3]
    )
    with pytest.raises(Exception) as e_info:
        assert template_builder._get_binning({})


def test_create_histograms(tmpdir):
    # needs to be expanded into a proper test
    config = {"Samples": {}, "Regions": {}, "Systematics": {}}
    template_builder.create_histograms(config, tmpdir, method="uproot")
