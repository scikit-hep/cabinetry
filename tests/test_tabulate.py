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
    total_stdev = [[[5.0, 2.0], [5.0, 2.0]], [[1.0], [1.0]]]
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
            "region_1\nbin 1": "25.00 \u00B1 5.00",
            "region_1\nbin 2": "5.00 \u00B1 2.00",
            "region_2\nbin 1": "8.00 \u00B1 1.00",
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
    total_stdev = [[[6.45], [2.15], [8.60]]]
    data = [[160]]
    channels = model.config.channels

    yield_table = tabulate._yields_per_bin(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {"sample": "Background", "Signal Region\nbin 1": "150.00 \u00B1 6.45"},
        {"sample": "Signal", "Signal Region\nbin 1": "50.00 \u00B1 2.15"},
        {"sample": "total", "Signal Region\nbin 1": "200.00 \u00B1 8.60"},
        {"sample": "data", "Signal Region\nbin 1": "160.00"},
    ]


def test__save_tables(tmp_path):
    table_dict = {
        "yields_per_bin": [
            {
                "sample": "Background",
                "Signal_region\nbin 1": "111.00 \u00B1 9.50",
                "Signal_region\nbin 2": "116.00 \u00B1 10.00",
            },
            {
                "sample": "Signal",
                "Signal_region\nbin 1": "1.00 \u00B1 0.50",
                "Signal_region\nbin 2": "5.00 \u00B1 1.00",
            },
            {
                "sample": "total",
                "Signal_region\nbin 1": "112.00 \u00B1 10.00",
                "Signal_region\nbin 2": "121.00 \u00B1 11.00",
            },
            {
                "sample": "data",
                "Signal_region\nbin 1": "112.00",
                "Signal_region\nbin 2": "123.00",
            },
        ],
        "yields_per_channel": [
            {"sample": "Background", "Signal_region": "227.00 \u00B1 16.00"},
            {"sample": "Signal", "Signal_region": "6.00 \u00B1 1.00"},
            {"sample": "total", "Signal_region": "233.00 \u00B1 17.00"},
            {"sample": "data", "Signal_region": "235.00"},
        ],
    }

    tabulate._save_tables(table_dict, tmp_path, "abc", "simple")
    fname_bin = tmp_path / "yields_per_bin_abc.txt"
    fname_channel = tmp_path / "yields_per_channel_abc.txt"
    assert fname_bin.is_file()  # bin table saved
    assert fname_channel.is_file()  # channel table saved
    assert fname_bin.read_text() == (
        "sample      Signal_region    Signal_region\n"
        "            bin 1            bin 2\n"
        "----------  ---------------  ---------------\n"
        "Background  111.00 \u00B1 9.50    116.00 \u00B1 10.00\n"
        "Signal      1.00 \u00B1 0.50      5.00 \u00B1 1.00\n"
        "total       112.00 \u00B1 10.00   121.00 \u00B1 11.00\n"
        "data        112.00           123.00\n"
    )
    assert fname_channel.read_text() == (
        "sample      Signal_region\n"
        "----------  ---------------\n"
        "Background  227.00 \u00B1 16.00\n"
        "Signal      6.00 \u00B1 1.00\n"
        "total       233.00 \u00B1 17.00\n"
        "data        235.00\n"
    )

    # suffix for latex, modification of newlines in header
    tabulate._save_tables(table_dict, tmp_path, "abc", "latex")
    fname_bin = tmp_path / "yields_per_bin_abc.tex"
    assert fname_bin.is_file()
    assert fname_bin.read_text().split("&")[1] == " Signal\\_region, bin 1   "

    # unchanged suffix for other formats
    tabulate._save_tables(table_dict, tmp_path, "abc", "html")
    fname_bin = tmp_path / "yields_per_bin_abc.html"
    assert fname_bin.is_file()


def test__yields_per_channel(
    example_spec_multibin, example_spec_with_background, caplog
):
    caplog.set_level(logging.DEBUG)

    # multiple channels
    model = pyhf.Workspace(example_spec_multibin).model()
    yields = [[30], [8.0]]
    total_stdev = [[5.39, 5.39], [1.0, 1.0]]
    data = [43, 10]
    channels = model.config.channels
    label = "pre-fit"
    caplog.clear()

    yield_table = tabulate._yields_per_channel(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {
            "sample": "Signal",
            "region_1": "30.00 \u00B1 5.39",
            "region_2": "8.00 \u00B1 1.00",
        },
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
    total_stdev = [[6.45, 2.15, 8.60]]
    data = [160]
    channels = model.config.channels

    yield_table = tabulate._yields_per_channel(
        model, yields, total_stdev, data, channels, label
    )
    assert yield_table == [
        {"sample": "Background", "Signal Region": "150.00 \u00B1 6.45"},
        {"sample": "Signal", "Signal Region": "50.00 \u00B1 2.15"},
        {"sample": "total", "Signal Region": "200.00 \u00B1 8.60"},
        {"sample": "data", "Signal Region": "160.00"},
    ]


@mock.patch("cabinetry.tabulate._save_tables")
@mock.patch(
    "cabinetry.tabulate._yields_per_channel",
    return_value=[{"yields_per_channel": None}],
)
@mock.patch(
    "cabinetry.tabulate._yields_per_bin", return_value=[{"yields_per_bin": None}]
)
@mock.patch("cabinetry.model_utils._filter_channels", side_effect=[["SR"], [], ["SR"]])
@mock.patch("cabinetry.model_utils._data_per_channel", return_value=[[12.0]])
def test_yields(
    mock_data, mock_filter, mock_bin, mock_channel, mock_save, example_spec, caplog
):
    # the return values of the functions producing the actual tables are hardcoded above
    # to be able to check that they are correctly propagated to the output
    caplog.set_level(logging.DEBUG)
    model = pyhf.Workspace(example_spec).model()
    model_pred = model_utils.ModelPrediction(
        model, [[[10.0]]], [[[0.3], [0.3]]], [[0.3, 0.3]], "pred"
    )
    data = [12.0, 1.0]  # with auxdata to strip via mock

    table_dict = tabulate.yields(model_pred, data)
    assert mock_data.call_args_list == [((model, data), {})]
    assert mock_filter.call_args_list == [((model, None), {})]
    assert mock_bin.call_args_list == [
        ((model, [[[10.0]]], [[[0.3], [0.3]]], [[12.0]], ["SR"], "pred"), {})
    ]
    assert mock_channel.call_count == 0
    assert mock_save.call_count == 1
    assert table_dict == {"yields_per_bin": [{"yields_per_bin": None}]}

    # no table to produce (does not call _filter_channels)
    table_dict = tabulate.yields(model_pred, data, per_bin=False, per_channel=False)
    assert "requested neither yields per bin nor per channel, no table produced" in [
        rec.message for rec in caplog.records
    ]
    assert table_dict == {}
    caplog.clear()

    # no channels to include
    table_dict = tabulate.yields(model_pred, data, channels="abc")
    assert mock_filter.call_args == ((model, "abc"), {})
    assert mock_bin.call_count == 1  # one call from before, no new call
    assert mock_channel.call_count == 0
    assert table_dict == {}

    # yields per channel, not per bin, do not save tables
    table_dict = tabulate.yields(
        model_pred, data, per_bin=False, per_channel=True, save_tables=False
    )
    assert mock_bin.call_count == 1  # one call from before
    assert mock_channel.call_args_list == [
        ((model, [[10.0]], [[0.3, 0.3]], [12.0], ["SR"], "pred"), {})
    ]
    assert table_dict == {"yields_per_channel": [{"yields_per_channel": None}]}
    assert mock_save.call_count == 1  # one call from before, no new call
