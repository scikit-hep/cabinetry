import logging
import pathlib
from unittest import mock

import boost_histogram as bh
import numpy as np
import pytest

from cabinetry.templates import builder


def test__ntuple_paths(caplog):
    caplog.set_level(logging.DEBUG)

    # only general path, no override
    assert builder._ntuple_paths("path.root", {}, {}, {}, None) == [
        pathlib.Path("path.root")
    ]

    # general path with region and sample templates
    assert builder._ntuple_paths(
        "{RegionPath}/{SamplePath}",
        {"RegionPath": "region"},
        {"SamplePath": "sample.root"},
        {},
        None,
    ) == [pathlib.Path("region/sample.root")]

    # SamplePath with list of two samples
    assert builder._ntuple_paths(
        "{RegionPath}/{SamplePath}",
        {"RegionPath": "region"},
        {"SamplePath": ["sample.root", "new.root"]},
        {},
        None,
    ) == [pathlib.Path("region/sample.root"), pathlib.Path("region/new.root")]

    # systematic with override for RegionPath and SamplePath
    assert builder._ntuple_paths(
        "{RegionPath}/{SamplePath}",
        {"RegionPath": "reg_1"},
        {"SamplePath": "path.root"},
        {
            "Name": "variation",
            "Up": {"SamplePath": ["variation.root", "new.root"], "RegionPath": "reg_2"},
        },
        "Up",
    ) == [pathlib.Path("reg_2/variation.root"), pathlib.Path("reg_2/new.root")]

    # systematic without override
    assert builder._ntuple_paths(
        "{SamplePath}", {}, {"SamplePath": "path.root"}, {"Name": "variation"}, "Up"
    ) == [pathlib.Path("path.root")]

    caplog.clear()

    # warning: no region path in template
    assert builder._ntuple_paths(
        "path.root", {"RegionPath": "region.root"}, {}, {}, None
    ) == [pathlib.Path("path.root")]
    assert "region override specified, but {RegionPath} not found in default path" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # warning: no sample path in template
    assert builder._ntuple_paths(
        "path.root", {}, {"SamplePath": "sample.root"}, {}, None
    ) == [pathlib.Path("path.root")]
    assert "sample override specified, but {SamplePath} not found in default path" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # error: no override for {RegionPath}
    with pytest.raises(ValueError, match="no path setting found for region region"):
        builder._ntuple_paths("{RegionPath}", {"Name": "region"}, {}, {}, None)

    # error: no override for {SamplePath}
    with pytest.raises(ValueError, match="no path setting found for sample sample"):
        builder._ntuple_paths("{SamplePath}", {}, {"Name": "sample"}, {}, None)


def test__variable():
    # no override
    assert builder._variable({"Variable": "jet_pt"}, {}, {}, None) == "jet_pt"

    # systematic with override
    assert (
        builder._variable(
            {"Variable": "jet_pt"},
            {},
            {"Name": "variation", "Up": {"Variable": "jet_pt_up"}},
            "Up",
        )
        == "jet_pt_up"
    )

    # systematic without override
    assert (
        builder._variable({"Variable": "jet_pt"}, {}, {"Name": "variation"}, "Up")
        == "jet_pt"
    )


def test__filter():
    # no override
    assert builder._filter({"Filter": "jet_pt > 0"}, {}, {}, None) == "jet_pt > 0"

    # no filter
    assert builder._filter({}, {}, {}, None) is None

    # systematic with override
    assert (
        builder._filter(
            {"Filter": "jet_pt > 0"},
            {},
            {"Name": "variation", "Up": {"Filter": "jet_pt > 100"}},
            "Up",
        )
        == "jet_pt > 100"
    )

    # systematic without override
    assert (
        builder._filter({"Filter": "jet_pt > 0"}, {}, {"Name": "variation"}, "Up")
        == "jet_pt > 0"
    )

    # sample-specific override
    assert (
        builder._filter({"Filter": "jet_pt > 0"}, {"Filter": "jet_pt > 100"}, {}, None)
        == "jet_pt > 100"
    )

    # sample-specific override, again overridden by systematic
    assert (
        builder._filter(
            {"Filter": "jet_pt > 0"},
            {"Filter": "jet_pt > 100"},
            {"Name": "variation", "Up": {"Filter": "jet_pt > 200"}},
            "Up",
        )
        == "jet_pt > 200"
    )


def test__weight():
    # no override
    assert builder._weight({}, {"Weight": "weight_mc"}, {}, None) == "weight_mc"

    # no weight
    assert builder._weight({}, {}, {}, None) is None

    # systematic with override
    assert (
        builder._weight(
            {},
            {"Weight": "weight_mc"},
            {"Name": "variation", "Up": {"Weight": "weight_modified"}},
            "Up",
        )
        == "weight_modified"
    )

    # systematic without override
    assert (
        builder._weight({}, {"Weight": "weight_mc"}, {"Name": "variation"}, "Up")
        == "weight_mc"
    )


def test__position_in_file():
    # no override
    assert builder._position_in_file({"Tree": "tree_name"}, {}, None) == "tree_name"

    # systematic with override
    assert (
        builder._position_in_file(
            {"Tree": "nominal"}, {"Name": "variation", "Up": {"Tree": "up_tree"}}, "Up"
        )
        == "up_tree"
    )

    # systematic without override
    assert (
        builder._position_in_file({"Tree": "nominal"}, {"Name": "variation"}, "Up")
        == "nominal"
    )


def test__binning():
    np.testing.assert_equal(builder._binning({"Binning": [1, 2, 3]}), [1, 2, 3])
    with pytest.raises(NotImplementedError, match="cannot determine binning"):
        builder._binning({})


def test__Builder():
    builder_instance = builder._Builder(pathlib.Path("path"), "file.root", "uproot")
    assert builder_instance.histogram_folder == pathlib.Path("path")
    assert builder_instance.general_path == "file.root"
    assert builder_instance.method == "uproot"


@mock.patch("cabinetry.templates.utils._name_and_save")
@mock.patch("cabinetry.histo.Histogram", return_value="cabinetry_histogram")
@mock.patch("cabinetry.contrib.histogram_creator.with_uproot", return_value="histogram")
@mock.patch(
    "cabinetry.templates.builder._ntuple_paths",
    return_value=[pathlib.Path("path_to_ntuple")],
)
def test__Builder_create_histogram(
    mock_path, mock_uproot_builder, mock_histo, mock_save
):
    histogram_folder = pathlib.Path("path")
    general_path = "{SamplePath}"
    # the binning [0] is not a proper binning, but simplifies the comparison
    region = {"Name": "test_region", "Variable": "x", "Binning": [0], "Filter": "x>3"}
    sample = {
        "Name": "sample",
        "Tree": "tree",
        "SamplePath": "path_to_sample",
        "Weight": "weight_mc",
    }
    systematic = {}
    template = "Up"

    builder_instance = builder._Builder(histogram_folder, general_path, "uproot")
    builder_instance._create_histogram(region, sample, systematic, template)

    # call to path creation
    assert mock_path.call_args_list == [
        ((general_path, region, sample, systematic, template), {})
    ]

    # verify the backend call happened properly
    assert mock_uproot_builder.call_args_list == [
        (
            ([pathlib.Path("path_to_ntuple")], "tree", "x", [0]),
            {"weight": "weight_mc", "selection_filter": "x>3"},
        )
    ]

    # conversion from bh.Histogram to cabinetry Histogram
    assert mock_histo.call_args_list == [(("histogram",), {})]

    # verify the call for saving wrapped histogram
    assert mock_save.call_args_list == [
        (
            (
                histogram_folder,
                "cabinetry_histogram",
                region,
                sample,
                systematic,
                template,
            ),
            {},
        )
    ]

    # other backends
    builder_unknown = builder._Builder(histogram_folder, "{SamplePath}", "unknown")
    with pytest.raises(NotImplementedError, match="unknown backend unknown"):
        builder_unknown._create_histogram(region, sample, systematic, template)


@mock.patch("cabinetry.templates.utils._name_and_save")
def test__Builder__wrap_custom_template_builder(mock_save):
    histogram = bh.Histogram(bh.axis.Variable([0, 1]))
    region = {"Name": "test_region"}
    sample = {"Name": "sample"}
    systematic = {}
    histogram_folder = pathlib.Path("path")

    def test_func(reg, sam, sys, tem):
        return histogram

    builder_instance = builder._Builder(histogram_folder, "file.root", "uproot")
    wrapped_func = builder_instance._wrap_custom_template_builder(test_func)

    # check the behavior of the wrapped function
    # when called, it should save the returned histogram
    wrapped_func(region, sample, systematic, "Up")

    assert mock_save.call_args_list == [
        ((histogram_folder, histogram, region, sample, systematic, "Up"), {})
    ]

    # wrapped function returns wrong type
    def test_func_wrong_return(reg, sam, sys, tem):
        return None

    wrapped_func_wrong_return = builder_instance._wrap_custom_template_builder(
        test_func_wrong_return
    )
    with pytest.raises(TypeError, match="must return a boost_histogram.Histogram"):
        wrapped_func_wrong_return(region, sample, systematic, "Up")
