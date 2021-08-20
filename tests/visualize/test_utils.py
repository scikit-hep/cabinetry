import pathlib
from unittest import mock

import matplotlib.pyplot as plt

from cabinetry.visualize import utils


def test__save_and_close(tmp_path):
    fig = plt.figure()
    fname = tmp_path / "fig.pdf"
    utils._save_and_close(fig, fname, False)
    assert fname.is_file()  # file was saved
    assert len(plt.get_fignums()) == 1  # figure is open

    fig = plt.figure()  # new figure to test closing
    utils._save_and_close(fig, fname, True)
    assert len(plt.get_fignums()) == 1  # previous figure still open
    plt.close("all")

    fig = mock.MagicMock()
    utils._save_and_close(fig, None, True)
    assert not fig.savefig.called  # no path provided, so figure was not saved


def test__log_figure_path():
    path = pathlib.Path("path/to_file.pdf")
    assert utils._log_figure_path(path) == pathlib.Path("path/to_file_log.pdf")

    # input is None
    path = None
    assert utils._log_figure_path(path) is None
