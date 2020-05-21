from pathlib import Path
import pytest

from cabinetry import template_builder


def test__get_ntuple_path():
    assert template_builder._get_ntuple_path(
        {"Path": "test/path.root"}, {}, {}
    ) == Path("test/path.root")


def test__get_selection():
    assert template_builder._get_selection({}, {"Variable": "jet_pt"}, {}) == "jet_pt"


def test__get_weight():
    assert template_builder._get_weight({"Weight": "weight_mc"}, {}, {}) == "weight_mc"
    assert template_builder._get_weight({}, {}, {}) == None


def test__get_position_in_file():
    assert template_builder._get_position_in_file({"Tree": "tree_name"}) == "tree_name"


def test__get_binning():
    assert template_builder._get_binning({"Binning": [1, 2, 3]}) == [1, 2, 3]
    with pytest.raises(Exception) as e_info:
        assert template_builder._get_binning({})


def test_create_histograms(tmpdir):
    # needs to be expanded into a proper test
    config = {"Samples": {}, "Regions": {}, "Systematics": {}}
    template_builder.create_histograms(
        config, tmpdir, method="uproot", only_nominal=True
    )
