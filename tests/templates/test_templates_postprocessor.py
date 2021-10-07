import logging
import pathlib
from unittest import mock

import numpy as np
import pytest

from cabinetry import histo
from cabinetry.templates import postprocessor


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
    postprocessor._fix_stat_unc(test_histo, name)
    assert np.allclose(test_histo.stdev, fixed_stdev)


@mock.patch("cabinetry.smooth.smooth_353qh_twice", return_value=np.asarray([1, 1.3]))
def test__apply_353qh_twice(mock_smooth):
    var = histo.Histogram.from_arrays([1, 2, 3], [1, 1.5], [0.1, 0.1])
    nom = histo.Histogram.from_arrays([1, 2, 3], [1, 1.2], [0.1, 0.1])

    postprocessor._apply_353qh_twice(var, nom, "abc")

    assert np.allclose(nom.yields, [1, 1.2])  # nominal unchanged
    # result of smoothing is [1, 1.3], multiplied by [1, 1.2] -> [1, 1.56]
    # [1, 1.56] scaled to integral 2.5 results in [0.9765625, 1.5234375]
    assert np.allclose(var.yields, [0.9765625, 1.5234375])


def test__smoothing_algorithm():
    reg = {"Name": "reg"}
    sam = {"Name": "sam"}
    sys = {"Name": "sys"}

    assert postprocessor._smoothing_algorithm(reg, sam, sys) is None

    sys = {"Name": "sys", "Smoothing": {"Algorithm": "abc"}}
    assert postprocessor._smoothing_algorithm(reg, sam, sys) == "abc"

    # region is missing in setting
    sys = {"Name": "sys", "Smoothing": {"Algorithm": "abc", "Regions": "r"}}
    assert postprocessor._smoothing_algorithm(reg, sam, sys) is None

    # region is included in setting
    sys = {"Name": "sys", "Smoothing": {"Algorithm": "abc", "Regions": ["r", "reg"]}}
    assert postprocessor._smoothing_algorithm(reg, sam, sys) == "abc"

    # sample is missing in setting
    sys = {"Name": "sys", "Smoothing": {"Algorithm": "abc", "Samples": "s"}}
    assert postprocessor._smoothing_algorithm(reg, sam, sys) is None

    # sample is included in setting
    sys = {"Name": "sys", "Smoothing": {"Algorithm": "abc", "Samples": ["s", "sam"]}}
    assert postprocessor._smoothing_algorithm(reg, sam, sys) == "abc"


@mock.patch("cabinetry.templates.postprocessor._apply_353qh_twice")
@mock.patch("cabinetry.templates.postprocessor._fix_stat_unc")
def test_apply_postprocessing(mock_stat, mock_smooth, caplog):
    caplog.set_level(logging.DEBUG)
    histogram = histo.Histogram.from_arrays([1, 2, 3], [1, 1], [float("nan"), 0.2])
    nom_hist = histo.Histogram.from_arrays([1, 2, 3], [2, 2], [0.1, 0.2])
    name = "test_histo"
    modified_histogram = postprocessor.apply_postprocessing(
        histogram, name, smoothing_algorithm="353QH, twice", nominal_histogram=nom_hist
    )

    # call to stat. unc. fix
    assert mock_stat.call_count == 1
    assert np.allclose(mock_stat.call_args[0][0].yields, histogram.yields)
    assert np.allclose(mock_stat.call_args[0][0].stdev, histogram.stdev, equal_nan=True)
    assert mock_stat.call_args[0][1] == name
    assert mock_stat.call_args[1] == {}

    # call to smoothing
    assert mock_smooth.call_count == 1
    assert np.allclose(mock_smooth.call_args[0][0].yields, histogram.yields)
    assert np.allclose(
        mock_smooth.call_args[0][0].stdev, histogram.stdev, equal_nan=True
    )
    assert np.allclose(mock_smooth.call_args[0][1].yields, nom_hist.yields)
    assert np.allclose(mock_smooth.call_args[0][1].stdev, nom_hist.stdev)
    assert mock_smooth.call_args[0][2] == name
    assert mock_smooth.call_args[1] == {}

    # the original histogram should be unchanged
    assert np.allclose(histogram.yields, [1, 1])
    assert np.allclose(histogram.stdev, [float("nan"), 0.2], equal_nan=True)
    # original and modified histogram should match (modifications were mocked out)
    assert np.allclose(histogram.yields, modified_histogram.yields)
    assert np.allclose(histogram.stdev, modified_histogram.stdev, equal_nan=True)

    # unknown smoothing algorithm
    _ = postprocessor.apply_postprocessing(
        histogram, name, smoothing_algorithm="abc", nominal_histogram=nom_hist
    )
    assert mock_stat.call_count == 2
    assert mock_smooth.call_count == 1
    assert "unknown smoothing algorithm abc" in [rec.message for rec in caplog.records]
    caplog.clear()

    # no smoothing
    _ = postprocessor.apply_postprocessing(
        histogram, name, smoothing_algorithm=None, nominal_histogram=None
    )
    assert mock_stat.call_count == 3
    assert mock_smooth.call_count == 1

    # known smoothing, but no nominal histogram
    with pytest.raises(
        ValueError, match="cannot apply smoothing, nominal histogram missing"
    ):
        _ = postprocessor.apply_postprocessing(
            histogram, name, smoothing_algorithm="353QH, twice", nominal_histogram=None
        )


@mock.patch("cabinetry.histo.name", return_value="histo_name")
def test__postprocessor(mock_name):
    processor = postprocessor._postprocessor(pathlib.Path("path"))

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
            "cabinetry.templates.postprocessor.apply_postprocessing",
            return_value=mock_new_histogram,
        ) as mock_postprocessing:
            # execute the provided function
            with mock.patch(
                "cabinetry.templates.postprocessor._smoothing_algorithm",
                return_value=None,
            ) as mock_smooth:
                processor(region, sample, systematic, template)
                assert mock_smooth.call_args_list == [
                    ((region, sample, systematic), {})
                ]

            # check that the relevant functions were called
            assert mock_from_config.call_args_list == [
                (
                    (pathlib.Path("path"), region, sample, systematic),
                    {"template": template, "modified": False},
                )
            ]  # histogram was created

            assert mock_postprocessing.call_args_list == [
                (
                    (mock_original_histogram, "histo_name"),
                    {"smoothing_algorithm": None, "nominal_histogram": None},
                )
            ]  # postprocessing was executed

            assert mock_new_histogram.save.call_args_list == [
                ((pathlib.Path("path/histo_name_modified"),), {})
            ]  # new histogram was saved (meaning right name was obtained)

            # postprocessor with smoothing
            systematic = {
                "Name": "systematic",
                "Smoothing": {"Algorithm": "353QH, twice"},
            }
            with mock.patch(
                "cabinetry.templates.postprocessor._smoothing_algorithm",
                return_value="353QH, twice",
            ) as mock_smooth:
                processor(region, sample, systematic, template)
                assert mock_smooth.call_args_list == [
                    ((region, sample, systematic), {})
                ]

            # nominal histogram was read
            assert mock_from_config.call_args[0][3] == {}

            # postprocessing called with smoothing algorithm and nominal histogram
            assert mock_postprocessing.call_args == (
                (mock_original_histogram, "histo_name"),
                {
                    "smoothing_algorithm": "353QH, twice",
                    "nominal_histogram": mock_original_histogram,
                },
            )
