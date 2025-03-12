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


def test__find_key_in_nested_dict():
    # key found at top level
    assert utils._find_key_in_nested_dict({"key": "value"}, "key") == "value"

    # key found in nested dict
    assert (
        utils._find_key_in_nested_dict({"key": {"nested_key": "value"}}, "nested_key")
        == "value"
    )

    # Key found in deeply nested dict
    assert (
        utils._find_key_in_nested_dict({"a": {"b": {"c": {"d": "value"}}}}, "d")
        == "value"
    )

    # Key found in one of multiple nested dicts
    assert (
        utils._find_key_in_nested_dict({"a": {"b": 1}, "c": {"d": "value"}}, "d")
        == "value"
    )

    # Stops at first match (shadowing test)
    assert (
        utils._find_key_in_nested_dict({"key": 1, "nested_key": {"key": 2}}, "key") == 1
    )

    # key not found
    assert utils._find_key_in_nested_dict({"key": "value"}, "not_key") is None
    assert utils._find_key_in_nested_dict({}, "key") is None
    assert (
        utils._find_key_in_nested_dict({"key": {"nested_key": "value"}}, "not_key")
        is None
    )
    assert utils._find_key_in_nested_dict({"key": {}}, "nested_key") is None
    assert utils._find_key_in_nested_dict({}, "not_key") is None
    #  Key inside a list should not be found in implementation
    assert (
        utils._find_key_in_nested_dict({"key_1": [{"key_2": "value"}]}, "key_2") is None
    )
