import numpy as np
import pyhf
import pytest

from cabinetry import tabulate


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("abc", 0), "abc\nbin 1"),
        (("abc", 2), "\nbin 3"),
    ],
)
def test__header_name(test_input, expected):
    assert tabulate._header_name(*test_input) == expected


def test__yields(example_spec_multibin, example_spec_with_background):
    # multiple channels
    model = pyhf.Workspace(example_spec_multibin).model()
    yields = [np.asarray([[25.0, 5.0]]), np.asarray([[8.0]])]
    total_stdev = [[5.0, 2.0], [1.0]]
    data = [np.asarray([35, 8]), np.asarray([10])]

    yield_table = tabulate._yields(model, yields, total_stdev, data)
    assert yield_table == [
        {
            "sample": "Signal",
            "region_1\nbin 1": "25.00",
            "\nbin 2": "5.00",
            "region_2\nbin 1": "8.00",
        },
        {
            "sample": "total",
            "region_1\nbin 1": "25.00 \u00B1 5.00",
            "\nbin 2": "5.00 \u00B1 2.00",
            "region_2\nbin 1": "8.00 \u00B1 1.00",
        },
        {
            "sample": "data",
            "region_1\nbin 1": "35.00",
            "\nbin 2": "8.00",
            "region_2\nbin 1": "10.00",
        },
    ]

    # multiple samples
    model = pyhf.Workspace(example_spec_with_background).model()
    yields = [np.asarray([[150.0], [50.0]])]
    total_stdev = [[8.60]]
    data = [np.asarray([160])]

    yield_table = tabulate._yields(model, yields, total_stdev, data)
    assert yield_table == [
        {"sample": "Background", "Signal Region\nbin 1": "150.00"},
        {"sample": "Signal", "Signal Region\nbin 1": "50.00"},
        {"sample": "total", "Signal Region\nbin 1": "200.00 \u00B1 8.60"},
        {"sample": "data", "Signal Region\nbin 1": "160.00"},
    ]
