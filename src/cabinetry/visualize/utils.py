"""Contains visualization utilities."""

import logging
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)


def _save_figure(
    fig: mpl.figure.Figure, path: pathlib.Path, close_figure: bool = False
) -> None:
    """Saves a figure at a given location and optionally closes it.

    Args:
        fig (matplotlib.figure.Figure): figure to save
        path (pathlib.Path): path where figure should be saved
        close_figure (bool, optional): whether to close figure after saving, defaults to
            False
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"saving figure as {path}")
    fig.savefig(path)
    if close_figure:
        plt.close(fig)
