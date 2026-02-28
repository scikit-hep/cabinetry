"""Provides utilities for template histogram handling."""

import pathlib
from typing import Any, Literal

from cabinetry import histo


def _check_for_override(
    systematic: dict[str, Any], template: Literal["Up", "Down"], option: str
) -> str | list[str] | None:
    """Returns an override if specified by a template of a systematic.

    Given a systematic and a string specifying which template is currently under
    consideration, check whether the systematic defines an override for an option.
    Return the override if it exists, otherwise return None.

    Args:
        systematic (dict[str, Any]): containing all systematic information
        template (Literal["Up", "Down"]): template considered: "Up" or "Down"
        option (str): the option for which the presence of an override is checked

    Returns:
        str | list[str] | None: either None if no override exists, or the override
    """
    return systematic.get(template, {}).get(option, None)


def _name_and_save(
    histogram_folder: pathlib.Path,
    histogram: histo.Histogram,
    region: dict[str, Any],
    sample: dict[str, Any],
    systematic: dict[str, Any],
    template: Literal["Up", "Down"] | None,
) -> None:
    """Generates a unique name for a histogram and saves the histogram.

    Args:
        histogram_folder (pathlib.Path): folder to save the histograms to
        histogram (histo.Histogram): histogram to save
        region (dict[str, Any]): containing all region information
        sample (dict[str, Any]): containing all sample information
        systematic (dict[str, Any]): containing all systematic information
        template (Literal["Up", "Down"] | None): template considered: "Up", "Down", or
            None for nominal
    """
    # generate a name for the histogram
    histogram_name = histo.name(region, sample, systematic, template=template)

    # check the histogram for common issues
    histogram.validate(histogram_name)

    # save it
    histo_path = histogram_folder / histogram_name
    histogram.save(histo_path)
