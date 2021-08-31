import logging
import pathlib
from unittest import mock

import boost_histogram as bh
import numpy as np
import pytest

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

    # override is a list
    assert template_builder._check_for_override(
        {"Up": {"setting": ["val", "val2"]}}, "Up", "setting"
    ) == ["val", "val2"]


def test__ntuple_paths(caplog):
    # only general path, no override
    assert template_builder._ntuple_paths("path.root", {}, {}, {}, None) == [
        pathlib.Path("path.root")
    ]

    # general path with region and sample templates
    assert (
        template_builder._ntuple_paths(
            "{RegionPath}/{SamplePaths}",
            {"RegionPath": "region"},
            {"SamplePaths": "sample.root"},
            {},
            None,
        )
        == [pathlib.Path("region/sample.root")]
    )

    # two SamplePaths
    assert (
        template_builder._ntuple_paths(
            "{RegionPath}/{SamplePaths}",
            {"RegionPath": "region"},
            {"SamplePaths": ["sample.root", "new.root"]},
            {},
            None,
        )
        == [pathlib.Path("region/sample.root"), pathlib.Path("region/new.root")]
    )

    # systematic with override for RegionPath and SamplePaths
    assert (
        template_builder._ntuple_paths(
            "{RegionPath}/{SamplePaths}",
            {"RegionPath": "reg_1"},
            {"SamplePaths": "path.root"},
            {
                "Name": "variation",
                "Up": {
                    "SamplePaths": ["variation.root", "new.root"],
                    "RegionPath": "reg_2",
                },
            },
            "Up",
        )
        == [pathlib.Path("reg_2/variation.root"), pathlib.Path("reg_2/new.root")]
    )

    # systematic without override
    assert template_builder._ntuple_paths(
        "{SamplePaths}", {}, {"SamplePaths": "path.root"}, {"Name": "variation"}, "Up"
    ) == [pathlib.Path("path.root")]

    caplog.set_level(logging.DEBUG)
    caplog.clear()

    # warning: no region path in template
    assert template_builder._ntuple_paths(
        "path.root", {"RegionPath": "region.root"}, {}, {}, None
    ) == [pathlib.Path("path.root")]
    assert "region override specified, but {RegionPath} not found in default path" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # warning: no region path in template
    assert template_builder._ntuple_paths(
        "path.root", {}, {"SamplePaths": "sample.root"}, {}, None
    ) == [pathlib.Path("path.root")]
    assert "sample override specified, but {SamplePaths} not found in default path" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # error: no override for {RegionPath}
    with pytest.raises(ValueError, match="no path setting found for region region"):
        template_builder._ntuple_paths("{RegionPath}", {"Name": "region"}, {}, {}, None)

    # error: no override for {SamplePaths}
    with pytest.raises(ValueError, match="no path setting found for sample sample"):
        template_builder._ntuple_paths(
            "{SamplePaths}", {}, {"Name": "sample"}, {}, None
        )


def test__variable():
    # no override
    assert template_builder._variable({"Variable": "jet_pt"}, {}, {}, None) == "jet_pt"

    # systematic with override
    assert (
        template_builder._variable(
            {"Variable": "jet_pt"},
            {},
            {"Name": "variation", "Up": {"Variable": "jet_pt_up"}},
            "Up",
        )
        == "jet_pt_up"
    )

    # systematic without override
    assert (
        template_builder._variable(
            {"Variable": "jet_pt"}, {}, {"Name": "variation"}, "Up"
        )
        == "jet_pt"
    )


def test__filter():
    # no override
    assert (
        template_builder._filter({"Filter": "jet_pt > 0"}, {}, {}, None) == "jet_pt > 0"
    )

    # no filter
    assert template_builder._filter({}, {}, {}, None) is None

    # systematic with override
    assert (
        template_builder._filter(
            {"Filter": "jet_pt > 0"},
            {},
            {"Name": "variation", "Up": {"Filter": "jet_pt > 100"}},
            "Up",
        )
        == "jet_pt > 100"
    )

    # systematic without override
    assert (
        template_builder._filter(
            {"Filter": "jet_pt > 0"}, {}, {"Name": "variation"}, "Up"
        )
        == "jet_pt > 0"
    )


def test__weight():
    # no override
    assert (
        template_builder._weight({}, {"Weight": "weight_mc"}, {}, None) == "weight_mc"
    )

    # no weight
    assert template_builder._weight({}, {}, {}, None) is None

    # systematic with override
    assert (
        template_builder._weight(
            {},
            {"Weight": "weight_mc"},
            {"Name": "variation", "Up": {"Weight": "weight_modified"}},
            "Up",
        )
        == "weight_modified"
    )

    # systematic without override
    assert (
        template_builder._weight(
            {}, {"Weight": "weight_mc"}, {"Name": "variation"}, "Up"
        )
        == "weight_mc"
    )


def test__position_in_file():
    # no override
    assert (
        template_builder._position_in_file({"Tree": "tree_name"}, {}, None)
        == "tree_name"
    )

    # systematic with override
    assert (
        template_builder._position_in_file(
            {"Tree": "nominal"}, {"Name": "variation", "Up": {"Tree": "up_tree"}}, "Up"
        )
        == "up_tree"
    )

    # systematic without override
    assert (
        template_builder._position_in_file(
            {"Tree": "nominal"}, {"Name": "variation"}, "Up"
        )
        == "nominal"
    )


def test__binning():
    np.testing.assert_equal(
        template_builder._binning({"Binning": [1, 2, 3]}), [1, 2, 3]
    )
    with pytest.raises(NotImplementedError, match="cannot determine binning"):
        template_builder._binning({})


def test__Builder():
    builder = template_builder._Builder(pathlib.Path("path"), "file.root", "uproot")
    assert builder.histogram_folder == pathlib.Path("path")
    assert builder.general_path == "file.root"
    assert builder.method == "uproot"


@mock.patch("cabinetry.template_builder._Builder._name_and_save")
@mock.patch("cabinetry.histo.Histogram.from_arrays", return_value="histogram")
@mock.patch(
    "cabinetry.contrib.histogram_creation.from_uproot", return_value=([1], [0.1])
)
def test__Builder_create_histogram(mock_uproot_builder, mock_histo, mock_save):
    # the binning [0] is not a proper binning, but simplifies the comparison
    region = {"Name": "test_region", "Variable": "x", "Binning": [0], "Filter": "x>3"}
    sample = {
        "Name": "sample",
        "Tree": "tree",
        "SamplePaths": "path_to_sample",
        "Weight": "weight_mc",
    }
    systematic = {}

    builder = template_builder._Builder(pathlib.Path("path"), "{SamplePaths}", "uproot")
    builder._create_histogram(region, sample, systematic, None)

    # verify the backend call happened properly
    assert mock_uproot_builder.call_args_list == [
        (
            ([pathlib.Path("path_to_sample")], "tree", "x", [0]),
            {"weight": "weight_mc", "selection_filter": "x>3"},
        )
    ]

    # verify the histogram conversion call
    assert mock_histo.call_args_list == [(([0], [1], [0.1]), {})]

    # verify the call for saving
    assert mock_save.call_args_list == [
        (("histogram", region, sample, systematic, None), {})
    ]

    # other backends
    builder_unknown = template_builder._Builder(
        pathlib.Path("path"), "{SamplePaths}", "unknown"
    )
    with pytest.raises(NotImplementedError, match="unknown backend unknown"):
        builder_unknown._create_histogram(region, sample, systematic, None)


@mock.patch("cabinetry.histo.name", return_value="name")
def test__Builder__name_and_save(mock_name):
    region = {"Name": "test_region"}
    sample = {"Name": "sample"}
    systematic = {}

    histogram = mock.MagicMock()

    builder = template_builder._Builder(pathlib.Path("path"), "file.root", "uproot")
    builder._name_and_save(histogram, region, sample, systematic, "Up")

    # check that the naming function was called, the histogram was validated and saved
    assert mock_name.call_args_list == [((region, sample, systematic, "Up"), {})]
    assert histogram.validate.call_args_list == [mock.call("name")]
    assert histogram.save.call_args_list == [mock.call(pathlib.Path("path/name"))]


@mock.patch("cabinetry.template_builder._Builder._name_and_save")
def test__Builder__wrap_custom_template_builder(mock_save):
    histogram = bh.Histogram(bh.axis.Variable([0, 1]))
    region = {"Name": "test_region"}
    sample = {"Name": "sample"}
    systematic = {}

    def test_func(reg, sam, sys, tem):
        return histogram

    builder = template_builder._Builder(pathlib.Path("path"), "file.root", "uproot")
    wrapped_func = builder._wrap_custom_template_builder(test_func)

    # check the behavior of the wrapped function
    # when called, it should save the returned histogram
    wrapped_func(region, sample, systematic, "Up")

    assert mock_save.call_args_list == [
        ((histogram, region, sample, systematic, "Up"), {})
    ]

    # wrapped function returns wrong type
    def test_func_wrong_return(reg, sam, sys, tem):
        return None

    wrapped_func_wrong_return = builder._wrap_custom_template_builder(
        test_func_wrong_return
    )
    with pytest.raises(TypeError, match="must return a boost_histogram.Histogram"):
        wrapped_func_wrong_return(region, sample, systematic, "Up")


def test_create_histograms():
    config = {"General": {"HistogramFolder": "path/", "InputPath": "file.root"}}
    method = "uproot"

    # no router
    with mock.patch("cabinetry.route.apply_to_all_templates") as mock_apply:
        template_builder.create_histograms(config, method)
        assert mock_apply.call_count == 1
        config_call, func_call = mock_apply.call_args[0]
        assert config_call == config
        assert (
            func_call.__name__ == "_create_histogram"
        )  # could also compare to function
        assert mock_apply.call_args[1] == {"match_func": None}

    # including a router
    mock_router = mock.MagicMock()
    with mock.patch("cabinetry.route.apply_to_all_templates") as mock_apply:
        template_builder.create_histograms(config, method, router=mock_router)

        # verify wrapper was set
        assert (
            mock_router.template_builder_wrapper.__name__
            == "_wrap_custom_template_builder"
        )

        assert mock_apply.call_count == 1
        config_call, func_call = mock_apply.call_args[0]
        assert config_call == config
        assert func_call.__name__ == "_create_histogram"
        assert mock_apply.call_args[1] == {
            "match_func": mock_router._find_template_builder_match
        }
