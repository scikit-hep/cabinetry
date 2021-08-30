import logging
from unittest import mock

import pyhf
import pytest

from cabinetry import model_utils
from cabinetry import tabulate


@pytest.mark.parametrize(
    "test_input, expected", [(("abc", 0), "abc\nbin 1"), (("abc", 2), "abc\nbin 3")]
)
def test__header_name(test_input, expected):
    assert tabulate._header_name(*test_input) == expected
    assert tabulate._header_name("abc", 2, unique=False) == "\nbin 3"


def test__yields_per_bin(example_spec_multibin, example_spec_with_background, caplog):
    caplog.set_level(logging.DEBUG)

    # multiple channels
    model = pyhf.Workspace(example_spec_multibin).model()
    yields = [[[25.0, 5.0]], [[8.0]]]
    total_stdev = [[5.0, 2.0], [1.0]]
    data = [[35, 8], [10]]
    channels = model.config.channels
    label = "pre-fit"
    caplog.clear()

    yield_table = tabulate._yields_per_bin(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {
            "sample": "Signal",
            "region_1\nbin 1": "25.00",
            "region_1\nbin 2": "5.00",
            "region_2\nbin 1": "8.00",
        },
        {
            "sample": "total",
            "region_1\nbin 1": "25.00 \u00B1 5.00",
            "region_1\nbin 2": "5.00 \u00B1 2.00",
            "region_2\nbin 1": "8.00 \u00B1 1.00",
        },
        {
            "sample": "data",
            "region_1\nbin 1": "35.00",
            "region_1\nbin 2": "8.00",
            "region_2\nbin 1": "10.00",
        },
    ]
    assert "yields per bin for pre-fit model prediction:" in caplog.records[0].message
    caplog.clear()

    # multiple samples
    model = pyhf.Workspace(example_spec_with_background).model()
    yields = [[[150.0], [50.0]]]
    total_stdev = [[8.60]]
    data = [[160]]
    channels = model.config.channels

    yield_table = tabulate._yields_per_bin(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {"sample": "Background", "Signal Region\nbin 1": "150.00"},
        {"sample": "Signal", "Signal Region\nbin 1": "50.00"},
        {"sample": "total", "Signal Region\nbin 1": "200.00 \u00B1 8.60"},
        {"sample": "data", "Signal Region\nbin 1": "160.00"},
    ]


def test__yields_per_channel(
    example_spec_multibin, example_spec_with_background, caplog
):
    caplog.set_level(logging.DEBUG)

    # multiple channels
    model = pyhf.Workspace(example_spec_multibin).model()
    yields = [[30], [8.0]]
    total_stdev = [5.39, 1.0]
    data = [43, 10]
    channels = model.config.channels
    label = "pre-fit"
    caplog.clear()

    yield_table = tabulate._yields_per_channel(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {"sample": "Signal", "region_1": "30.00", "region_2": "8.00"},
        {
            "sample": "total",
            "region_1": "30.00 \u00B1 5.39",
            "region_2": "8.00 \u00B1 1.00",
        },
        {"sample": "data", "region_1": "43.00", "region_2": "10.00"},
    ]
    assert (
        "yields per channel for pre-fit model prediction:" in caplog.records[0].message
    )
    caplog.clear()

    # multiple samples
    model = pyhf.Workspace(example_spec_with_background).model()
    yields = [[150.0, 50]]
    total_stdev = [8.60]
    data = [160]
    channels = model.config.channels

    yield_table = tabulate._yields_per_channel(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {"sample": "Background", "Signal Region": "150.00"},
        {"sample": "Signal", "Signal Region": "50.00"},
        {"sample": "total", "Signal Region": "200.00 \u00B1 8.60"},
        {"sample": "data", "Signal Region": "160.00"},
    ]


@mock.patch("cabinetry.tabulate._yields_per_channel")
@mock.patch("cabinetry.tabulate._yields_per_bin")
@mock.patch("cabinetry.model_utils._filter_channels", side_effect=[["SR"], [], ["SR"]])
@mock.patch("cabinetry.model_utils._data_per_channel", return_value=[[12.0]])
def test_yields(mock_data, mock_filter, mock_bin, mock_channel, example_spec, caplog):
    caplog.set_level(logging.DEBUG)
    model = pyhf.Workspace(example_spec).model()
    model_pred = model_utils.ModelPrediction(model, [[[10.0]]], [[0.3]], [0.3], "pred")
    data = [12.0, 1.0]  # with auxdata to strip via mock

    tabulate.yields(model_pred, data)
    assert mock_data.call_args_list == [((model, data), {})]
    assert mock_filter.call_args_list == [((model, None), {})]
    assert mock_bin.call_args_list == [
        ((model, [[[10.0]]], [[0.3]], [[12.0]], ["SR"], "pred"), {})
    ]
    assert mock_channel.call_count == 0

    # no table to produce (does not call _filter_channels)
    tabulate.yields(model_pred, data, per_bin=False, per_channel=False)
    assert "requested neither yields per bin nor per channel, no table produced" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # no channels to include
    tabulate.yields(model_pred, data, channels="abc")
    assert mock_filter.call_args == ((model, "abc"), {})
    assert mock_bin.call_count == 1  # one call from before, no new call
    assert mock_channel.call_count == 0

    # yields per channel, not per bin
    tabulate.yields(model_pred, data, per_bin=False, per_channel=True)
    assert mock_bin.call_count == 1  # one call from before
    assert mock_channel.call_args_list == [
        ((model, [[10.0]], [0.3], [12.0], ["SR"], "pred"), {})
    ]
