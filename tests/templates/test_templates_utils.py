import pathlib
from unittest import mock

from cabinetry.templates import utils


def test__check_for_override():
    # override exists for template
    assert (
        utils._check_for_override({"Up": {"setting": "val"}}, "Up", "setting") == "val"
    )

    # no override for template
    assert utils._check_for_override({}, "Up", "setting") is None

    # no option requested
    assert utils._check_for_override({"Up": {"setting": "val"}}, "Up", "") is None

    # override is a list
    assert utils._check_for_override(
        {"Up": {"setting": ["val", "val2"]}}, "Up", "setting"
    ) == ["val", "val2"]


@mock.patch("cabinetry.histo.name", return_value="name")
def test__name_and_save(mock_name):
    histogram_folder = pathlib.Path("path")
    histogram = mock.MagicMock()
    region = {"Name": "test_region"}
    sample = {"Name": "sample"}
    systematic = {}

    utils._name_and_save(histogram_folder, histogram, region, sample, systematic, "Up")

    # check that the naming function was called, the histogram was validated and saved
    assert mock_name.call_args_list == [
        ((region, sample, systematic), {"template": "Up"})
    ]
    assert histogram.validate.call_args_list == [mock.call("name")]
    assert histogram.save.call_args_list == [mock.call(pathlib.Path("path/name"))]
