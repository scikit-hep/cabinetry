import pytest

from cabinetry import table


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("abc", 0), "abc\nbin 1"),
        (("abc", 2), "\nbin 3"),
    ],
)
def test__header_name(test_input, expected):
    assert table._header_name(*test_input) == expected


def test__yields():
    ...
