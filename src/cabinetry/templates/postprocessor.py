"""Applies optional post-processing to template histograms."""

import copy
import logging
import pathlib
from typing import Any, Dict, Optional

import numpy as np

from cabinetry import configuration
from cabinetry import histo
from cabinetry import route
from cabinetry import smooth
from cabinetry._typing import Literal


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram: histo.Histogram, name: str) -> None:
    """Replaces NaN statistical uncertainties by zero for a histogram.

    Modifies the histogram handed over in the argument.

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to fix
        name (str): histogram name for logging
    """
    nan_pos = np.where(np.isnan(histogram.stdev))[0]
    if len(nan_pos) > 0:
        log.debug(f"fixing ill-defined stat. unc. for {name}")
        histogram.stdev = np.nan_to_num(histogram.stdev, nan=0.0)


def _apply_353qh_twice(
    variation: histo.Histogram, nominal: histo.Histogram, name: str
) -> None:
    """Smooths systematic template histogram with the "353QH, twice" algorithm.

    The algorithm is applied to the ratio systematic variation / nominal. The nominal
    histogram stays the same, while the variation histogram is modified. Statistical
    uncertainties do not change (and the algorithm is not aware of statistical
    uncertainties). The total yield of the variation histogram stays unchanged.

    Args:
        variation (histo.Histogram): histogram of systematic variation
        nominal (histo.Histogram): associated nominal histogram
        name (str): histogram name for logging
    """
    log.debug(f"applying smoothing to {name}")
    # smooth relative effect of systematic (systematic/nominal)
    smooth_var = (
        smooth.smooth_353qh_twice(variation.yields / nominal.yields) * nominal.yields
    )
    # scale to match original sum of yields of variation
    variation.yields = smooth_var * sum(variation.yields) / sum(smooth_var)


def _smoothing_algorithm(
    region: Dict[str, Any], sample: Dict[str, Any], systematic: Dict[str, Any]
) -> Optional[str]:
    """Returns name of algorithm to use for smoothing, or None otherwise.

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information

    Returns:
        Optional[str]: name of smoothing algorithm or None
    """
    smoothing = systematic.get("Smoothing", None)
    if smoothing is None:
        return None

    # check for region and sample restrictions specified for smoothing, apply to all
    # regions and samples by default if no further specification is made
    if not configuration._x_contains_y(region, smoothing, "Regions"):
        # regions are specified and current region does not match, do not smooth
        return None

    if not configuration._x_contains_y(sample, smoothing, "Samples"):
        # samples are specified and current sample does not match, do not smooth
        return None

    # smoothing algorithm needs to be applied
    smoothing_algorithm = smoothing["Algorithm"]
    return smoothing_algorithm


def apply_postprocessing(
    histogram: histo.Histogram,
    name: str,
    *,
    smoothing_algorithm: Optional[str] = None,
    nominal_histogram: Optional[histo.Histogram] = None,
) -> histo.Histogram:
    """Returns a new histogram with post-processing applied.

    The histogram handed to the function stays unchanged. A copy of the histogram
    receives post-processing and is then returned. The postprocessing consists of a fix
    for NaN statistical uncertainties and optional smoothing.

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to postprocess
        name (str): histogram name for logging
        smoothing_algorithm (Optional[str]): name of smoothing algorithm to apply,
            defaults to None (no smoothing done)
        nominal_histogram (Optional[cabinetry.histo.Histogram]): nominal histogram
            (needed for smoothing), defaults to None

    Returns:
        cabinetry.histo.Histogram: the histogram with post-processing applied
    """
    # copy histogram to new object to leave it unchanged
    modified_histogram = copy.deepcopy(histogram)
    _fix_stat_unc(modified_histogram, name)
    if smoothing_algorithm is not None:
        if smoothing_algorithm == "353QH, twice":
            if nominal_histogram is None:
                raise ValueError("cannot apply smoothing, nominal histogram missing")
            _apply_353qh_twice(modified_histogram, nominal_histogram, name)
        else:
            log.warning(f"unknown smoothing algorithm {smoothing_algorithm}")
    return modified_histogram


def _postprocessor(histogram_folder: pathlib.Path) -> route.ProcessorFunc:
    """Returns the post-processing function to be applied to template histograms.

    Needed by ``cabinetry.route.apply_to_all_templates``. Could alternatively create a
    ``Postprocessor`` class that contains processors.

    Args:
        histogram_folder (pathlib.Path): folder containing histograms

    Returns:
        route.ProcessorFunc: function to apply to a template histogram
    """

    def process_template(
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        template: Optional[Literal["Up", "Down"]],
    ) -> None:
        """Applies post-processing to a single histogram.

        Args:
            region (Dict[str, Any]): containing all region information
            sample (Dict[str, Any]): containing all sample information
            systematic (Dict[str, Any]): containing all systematic information
            template (Optional[Literal["Up", "Down"]]): template considered: "Up",
                "Down", or None for nominal
        """
        histogram = histo.Histogram.from_config(
            histogram_folder,
            region,
            sample,
            systematic,
            template=template,
            modified=False,
        )
        histogram_name = histo.name(region, sample, systematic, template=template)

        smoothing_algorithm = _smoothing_algorithm(region, sample, systematic)
        if smoothing_algorithm is None:
            nominal_histogram = None
        else:
            # to apply smoothing, the associated nominal histogram is needed (smoothing
            # can currently not be applied to nominal itself, but for systematics is
            # applied to the ratio variation / nominal)
            nominal_histogram = histo.Histogram.from_config(
                histogram_folder, region, sample, {}, modified=False
            )

        new_histogram = apply_postprocessing(
            histogram,
            histogram_name,
            smoothing_algorithm=smoothing_algorithm,
            nominal_histogram=nominal_histogram,
        )
        histogram.validate(histogram_name)
        new_histo_path = histogram_folder / (histogram_name + "_modified")
        new_histogram.save(new_histo_path)

    return process_template
