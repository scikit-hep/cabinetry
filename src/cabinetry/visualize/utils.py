"""Contains visualization utilities."""

import logging
import pathlib
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)


def _save_and_close(
    fig: mpl.figure.Figure, path: Optional[pathlib.Path], close_figure: bool
) -> None:
    """Saves a figure at a given location if path is provided and optionally closes it.

    Args:
        fig (matplotlib.figure.Figure): figure to save
        path (Optional[pathlib.Path]): path where figure should be saved, or None to not
            save it
        close_figure (bool): whether to close figure after saving
    """
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        log.debug(f"saving figure as {path}")
        fig.savefig(path)
    if close_figure:
        plt.close(fig)
