"""Provides visualization utilities."""

import fnmatch
import logging
import pathlib
from typing import List, Optional, Set, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

from cabinetry import fit


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


def _log_figure_path(path: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
    """Adds a suffix to a figure path to indicate the use of a logarithmic axis.

    If the path is None (since figure should not be saved), it will stay None.

    Args:
        path (Optional[pathlib.Path]): original path to figure, or None if figure should
            not be saved

    Returns:
        Optional[pathlib.Path]: new path to figure including _log suffix, or None if
            original path is None
    """
    if path is not None:
        return path.with_name(path.stem + "_log" + path.suffix)
    return None


def _exclude_matching(
    fit_results: fit.FitResults,
    *,
    exclude: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    exclude_by_type: Optional[List[str]] = None,
) -> Set[str]:

    labels = fit_results.labels
    types = fit_results.types

    if exclude is None:
        exclude_set = set()
    elif isinstance(exclude, str):
        exclude_set = set(fnmatch.filter(labels, exclude))
    elif isinstance(exclude, (list, tuple)):
        exclude_set = set().union(
            *[set(fnmatch.filter(labels, match_str)) for match_str in exclude]
        )
    else:
        raise TypeError("exclude must be a string, list, or tuple")

    # exclude by type
    if exclude_by_type is None:
        exclude_by_type = ["staterror"]

    exclude_set.update(
        [
            label
            for label, kinds in zip(labels, types)
            if bool(set(kinds) & set(exclude_by_type))
        ]
    )
    return exclude_set
