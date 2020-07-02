from unittest import mock

import pytest

from cabinetry import visualize
from cabinetry.contrib import histogram_drawing


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("SR", True), "SR_prefit.pdf"),
        (("SR", False), "SR_postfit.pdf"),
        (("SR 1", True), "SR-1_prefit.pdf"),
        (("SR 1", False), "SR-1_postfit.pdf"),
    ],
)
def test__build_figure_name(test_input, expected):
    assert visualize._build_figure_name(*test_input) == expected


@mock.patch("cabinetry.contrib.histogram_drawing.data_MC_matplotlib")
@mock.patch("cabinetry.visualize.histo.load_from_config", return_value=({}, ""))
def test_data_MC(mock_load, mock_draw, tmp_path):
    """contrib.histogram_drawing is only imported depending on the keyword argument,
    so the patch works differently from the patch of load_from_config
    Generally it seems like following the path to the module is preferred, but that
    does not work for the `data_MC_matplotlib` case. For some information see also
    https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    """
    config = {
        "Regions": [{"Name": "reg_1", "Variable": "x"}],
        "Samples": [{"Name": "sample_1"}],
    }

    visualize.data_MC(config, tmp_path, tmp_path, prefit=True, method="matplotlib")

    # the call_args_list contains calls (outer round brackets), first filled with
    # arguments (inner round brackets) and then keyword arguments
    assert mock_load.call_args_list == [
        (
            (
                tmp_path,
                {"Name": "sample_1"},
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "nominal"},
            ),
            {"modified": True},
        )
    ]
    assert mock_draw.call_args_list == [
        (
            (
                [{"label": "sample_1", "isData": False, "hist": {}, "variable": "x"}],
                tmp_path / "reg_1_prefit.pdf",
            ),
        )
    ]

    # other plotting method
    with pytest.raises(Exception) as e_info:
        visualize.data_MC(config, tmp_path, tmp_path, prefit=True, method="unknown")

    # postfit
    with pytest.raises(Exception) as e_info:
        visualize.data_MC(config, tmp_path, tmp_path, prefit=False, method="matplotlib")
