from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from cabinetry import visualize


MockHistogram = namedtuple("MockHistogram", ["bins", "yields", "stdev"])


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


@mock.patch("cabinetry.contrib.matplotlib_visualize.data_MC")
@mock.patch(
    "cabinetry.histo.Histogram.from_config", return_value=MockHistogram([], [], []),
)
def test_data_MC(mock_load, mock_draw, tmp_path):
    """contrib.matplotlib_visualize is only imported depending on the keyword argument,
    so cannot patch via cabinetry.visualize.matplotlib_visualize
    Generally it seems like following the path to the module is preferred, but that
    does not work for the `data_MC` case. For some information see also
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
                {"Name": "reg_1", "Variable": "x"},
                {"Name": "sample_1"},
                {"Name": "nominal"},
            ),
            {"modified": True},
        )
    ]
    assert mock_draw.call_args_list == [
        (
            (
                [
                    {
                        "label": "sample_1",
                        "isData": False,
                        "hist": {"bins": [], "yields": [], "stdev": []},
                        "variable": "x",
                    }
                ],
                tmp_path / "reg_1_prefit.pdf",
            ),
        )
    ]

    # other plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.data_MC(config, tmp_path, tmp_path, prefit=True, method="unknown")

    # postfit
    with pytest.raises(NotImplementedError, match="only prefit implemented so far"):
        visualize.data_MC(config, tmp_path, tmp_path, prefit=False, method="matplotlib")


@mock.patch("cabinetry.contrib.matplotlib_visualize.correlation_matrix")
def test_correlation_matrix(mock_draw):
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    labels = ["a", "b"]
    folder_path = "tmp"
    figure_path = Path(folder_path) / "correlation_matrix.pdf"

    visualize.correlation_matrix(corr_mat, labels, folder_path, method="matplotlib")

    assert mock_draw.call_args_list == [((corr_mat, labels, figure_path),)]

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.correlation_matrix(corr_mat, labels, folder_path, method="unknown")


@mock.patch("cabinetry.contrib.matplotlib_visualize.pulls")
def test_pulls(mock_draw):
    bestfit = np.asarray([0.8, 1.0, 1.1])
    uncertainty = np.asarray([0.9, 1.0, 0.7])
    labels = ["a", "b", "c"]
    exclude_list = ["a"]
    folder_path = "tmp"

    expected_bestfit = np.asarray([1.0, 1.1])
    expected_uncertainty = np.asarray([1.0, 0.7])
    expected_labels = np.asarray(["b", "c"])
    figure_path = Path(folder_path) / "pulls.pdf"

    visualize.pulls(
        bestfit, uncertainty, labels, exclude_list, folder_path, method="matplotlib"
    )

    mock_draw.assert_called_once()
    assert np.allclose(mock_draw.call_args[0][0], expected_bestfit)
    assert np.allclose(mock_draw.call_args[0][1], expected_uncertainty)
    assert np.any(
        [
            mock_draw.call_args[0][2][i] == expected_labels[i]
            for i in range(len(expected_labels))
        ]
    )
    assert mock_draw.call_args[0][3] == figure_path
    assert mock_draw.call_args[1] == {}

    # unknown plotting method
    with pytest.raises(NotImplementedError, match="unknown backend: unknown"):
        visualize.pulls(
            bestfit, uncertainty, labels, exclude_list, folder_path, method="unknown"
        )
