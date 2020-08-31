import pathlib
from unittest import mock

import numpy as np
import pytest

from cabinetry import histo
from cabinetry import template_postprocessor


@pytest.mark.parametrize(
    "test_histo, fixed_stdev",
    [
        (
            histo.Histogram.from_arrays([1, 2, 3], [1, 2], [float("nan"), 0.2]),
            [0.0, 0.2],
        ),
        (histo.Histogram.from_arrays([1, 2, 3], [1, 2], [0.1, 0.2]), [0.1, 0.2]),
    ],
)
def test__fix_stat_unc(test_histo, fixed_stdev):
    name = "test_histo"
    template_postprocessor._fix_stat_unc(test_histo, name)
    assert np.allclose(test_histo.stdev, fixed_stdev)


def test_apply_postprocessing():
    histogram = histo.Histogram.from_arrays([1, 2, 3], [1, 1], [float("nan"), 0.2])
    name = "test_histo"
    fixed_stdev = [0.0, 0.2]
    fixed_histogram = template_postprocessor.apply_postprocessing(histogram, name)
    assert np.allclose(fixed_histogram.stdev, fixed_stdev)
    # the original histogram should be unchanged
    np.testing.assert_equal(histogram.stdev, [float("nan"), 0.2])


@mock.patch(
    "cabinetry.template_postprocessor.apply_postprocessing",
    return_value="new_histogram",
)
@mock.patch("cabinetry.histo.build_name", return_value="histo_name")
def test__get_postprocessor(mock_name, mock_apply):
    postprocessor = template_postprocessor._get_postprocessor(pathlib.Path("path"))

    region = {"Name": "region"}
    sample = {"Name": "sample"}
    systematic = {"Name": "systematic"}
    template = "Up"

    mock_original_histogram = mock.MagicMock()
    mock_new_histogram = mock.MagicMock()
    with mock.patch(
        "cabinetry.histo.Histogram.from_config", return_value=mock_original_histogram
    ) as mock_from_config:
        with mock.patch(
            "cabinetry.template_postprocessor.apply_postprocessing",
            return_value=mock_new_histogram,
        ) as mock_postprocessing:
            # execute the provided function
            postprocessor(region, sample, systematic, template)

            # check that the relevant functions were called
            assert mock_from_config.call_args_list == [
                (
                    (pathlib.Path("path"), region, sample, systematic),
                    {"modified": False, "template": template},
                )
            ]  # histogram was created

            assert mock_postprocessing.call_args_list == [
                ((mock_original_histogram, "histo_name"), {})
            ]  # postprocessing was executed

            assert mock_new_histogram.save.call_args_list == [
                ((pathlib.Path("path/histo_name_modified"),), {})
            ]  # new histogram was saved (meaning right name was obtained)


@mock.patch("cabinetry.route.apply_to_all_templates")
@mock.patch("cabinetry.template_postprocessor._get_postprocessor", return_value="func")
def test_run(mock_postprocessor, mock_apply):
    config = {"General": {"HistogramFolder": "path/"}}
    template_postprocessor.run(config)

    assert mock_postprocessor.call_args_list == [((pathlib.Path("path/"),), {})]
    assert mock_apply.call_args_list == [((config, "func"), {})]
