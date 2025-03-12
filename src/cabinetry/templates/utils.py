"""Provides utilities for template histogram handling."""

import pathlib
from typing import Any, Dict, List, Literal, Optional, Union

from cabinetry import histo


def _check_for_override(
    systematic: Dict[str, Any], template: Literal["Up", "Down"], option: str
) -> Optional[Union[str, List[str]]]:
    """Returns an override if specified by a template of a systematic.

    Given a systematic and a string specifying which template is currently under
    consideration, check whether the systematic defines an override for an option.
    Return the override if it exists, otherwise return None.

    Args:
        systematic (Dict[str, Any]): containing all systematic information
        template (Literal["Up", "Down"]): template considered: "Up" or "Down"
        option (str): the option for which the presence of an override is checked

    Returns:
        Optional[Union[str, List[str]]]: either None if no override exists, or the
        override
    """
    return systematic.get(template, {}).get(option, None)


def _name_and_save(
    histogram_folder: pathlib.Path,
    histogram: histo.Histogram,
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> None:
    """Generates a unique name for a histogram and saves the histogram.

    Args:
        histogram_folder (pathlib.Path): folder to save the histograms to
        histogram (histo.Histogram): histogram to save
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic
        template (Optional[Literal["Up", "Down"]]): template considered: "Up",
            "Down", or None for nominal
    """
    # generate a name for the histogram
    histogram_name = histo.name(region, sample, systematic, template=template)

    # check the histogram for common issues
    histogram.validate(histogram_name)

    # save it
    histo_path = histogram_folder / histogram_name
    histogram.save(histo_path)


def _find_key_in_nested_dict(d: Dict[str, Any], target_key: str) -> Optional[Any]:
    """
    Searches for a key in a nested dictionary at any level and
    returns its value if found.

    Args:
        d (Dict[str, Any]): The nested dictionary to search.
        target_key (str): The key to search for.

    Returns:
        Optional[Any]: The value associated with the target_key
            if found, otherwise None.
    """
    stack = [d]  # Initialize stack with the root dictionary

    while stack:
        current = stack.pop()  # Pop the last dictionary from the stack

        if target_key in current:
            return current[target_key]  # Key found, return the value

        # Add nested dictionaries to the stack
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)

    return None  # Key not found
