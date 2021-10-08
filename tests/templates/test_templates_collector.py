import logging
import pathlib
from unittest import mock

import pytest

from cabinetry.templates import collector


def test__histo_path(caplog):
    caplog.set_level(logging.DEBUG)

    # only general path, no override
    assert collector._histo_path("path.root:h1", "", {}, {}, {}, None) == "path.root:h1"

    # general path with region, sample and nominal variation
    assert (
        collector._histo_path(
            "{RegionPath}.root:{SamplePath}_{VariationPath}",
            "nominal",
            {"RegionPath": "region"},
            {"SamplePath": "sample"},
            {},
            None,
        )
        == "region.root:sample_nominal"
    )

    # systematic with override for VariationPath
    assert (
        collector._histo_path(
            "{RegionPath}.root:{SamplePath}_{VariationPath}",
            "nominal",
            {"RegionPath": "reg_1"},
            {"SamplePath": "path"},
            {"Name": "variation", "Up": {"VariationPath": "up"}},
            "Up",
        )
        == "reg_1.root:path_up"
    )

    caplog.clear()

    # systematic without override, results in warning from VariationPath
    assert (
        collector._histo_path(
            "f.root:h1_{VariationPath}",
            "",
            {"Name": "reg"},
            {"Name": "sam"},
            {"Name": "variation"},
            "Up",
        )
        == "f.root:h1_"
    )
    assert "no VariationPath override specified for reg / sam / variation Up" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # warning: no region path in template
    assert (
        collector._histo_path(
            "f.root:h1", "", {"RegionPath": "region.root"}, {}, {}, None
        )
        == "f.root:h1"
    )
    assert "region override specified, but {RegionPath} not found in default path" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # warning: no sample path in template
    assert (
        collector._histo_path(
            "f.root:h1", "", {}, {"SamplePath": "sample.root"}, {}, None
        )
        == "f.root:h1"
    )
    assert "sample override specified, but {SamplePath} not found in default path" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # warning: no colon in path
    assert collector._histo_path("f.root", "", {}, {}, {}, None) == "f.root"
    assert "no colon found in path f.root, may not be able to find histogram" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # error: no override for {RegionPath}
    with pytest.raises(ValueError, match="no path setting found for region region"):
        collector._histo_path("{RegionPath}", "", {"Name": "region"}, {}, {}, None)

    # error: no override for {SamplePath}
    with pytest.raises(ValueError, match="no path setting found for sample sample"):
        collector._histo_path("{SamplePath}", "", {}, {"Name": "sample"}, {}, None)


@mock.patch("cabinetry.templates.utils._name_and_save")
@mock.patch("cabinetry.histo.Histogram", return_value="cabinetry_histogram")
@mock.patch("cabinetry.contrib.histogram_reader.with_uproot", return_value="histogram")
@mock.patch("cabinetry.templates.collector._histo_path", return_value="path_to_hist")
def test__collector(mock_path, mock_uproot_hist, mock_histo, mock_save):
    histogram_folder = pathlib.Path("path")
    general_path = "f.root:{variation_path}"
    variation_path = "nominal"
    processor = collector._collector(
        histogram_folder, general_path, variation_path, "uproot"
    )

    region = {"Name": "region"}
    sample = {"Name": "sample"}
    systematic = {"Name": "systematic"}
    template = "Up"

    # execute processor
    processor(region, sample, systematic, template)

    # call to path creation
    assert mock_path.call_args_list == [
        ((general_path, variation_path, region, sample, systematic, template), {})
    ]

    # call to uproot backend
    assert mock_uproot_hist.call_args_list == [(("path_to_hist",), {})]

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
    processor_unknown = collector._collector(
        histogram_folder, general_path, variation_path, "unknown"
    )
    with pytest.raises(NotImplementedError, match="unknown backend unknown"):
        processor_unknown(region, sample, systematic, None)
