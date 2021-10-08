"""Creates required template histograms from columnar data."""

import functools
import logging
import pathlib
from typing import Any, Dict, List, Optional

import boost_histogram as bh
import numpy as np

from cabinetry import configuration
from cabinetry import histo
from cabinetry import route
from cabinetry._typing import Literal
from cabinetry.templates import utils


log = logging.getLogger(__name__)


def _ntuple_paths(
    general_path: str,
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> List[pathlib.Path]:
    """Returns the paths to ntuples for a region-sample-systematic-template.

    A path is built starting from the path specified in the general options in the
    configuration file. This path can contain placeholders for region- and sample-
    specific overrides, via ``{Region}`` and ``{Sample}``. For non-nominal templates, it
    is possible to override the sample path if the ``SamplePath`` option is specified
    for the template. If ``SamplePath`` is a list, return a list of paths (one per
    entry in the list).

    Args:
        general_path (str): path specified in general settings, with sections that can
            be overridden by region / sample / systematic settings
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): template considered: "Up", "Down",
            or None for nominal

    Raises:
        ValueError: when ``RegionPath`` placeholder is used, but region setting is not
            specified
        ValueError: when ``SamplePath`` placeholder is used, but sample setting is not
            specified

    Returns:
        List[pathlib.Path]: list of paths to ntuples
    """
    # obtain region and sample paths, if they are defined
    region_path = region.get("RegionPath", None)
    sample_paths = sample.get("SamplePath", None)

    # check whether a systematic is being processed, and whether overrides exist
    if template is not None:
        # determine whether the template has an override for RegionPath specified
        region_override = utils._check_for_override(systematic, template, "RegionPath")
        if region_override is not None:
            region_path = region_override

        # check for SamplePath override
        sample_override = utils._check_for_override(systematic, template, "SamplePath")
        if sample_override is not None:
            sample_paths = sample_override

    region_template_exists = "{RegionPath}" in general_path
    if region_path is not None:
        if not region_template_exists:
            log.warning(
                "region override specified, but {RegionPath} not found in default path"
            )
        general_path = general_path.replace("{RegionPath}", region_path)
    elif region_template_exists:
        raise ValueError(f"no path setting found for region {region['Name']}")

    sample_template_exists = "{SamplePath}" in general_path
    if sample_paths is not None:
        if not sample_template_exists:
            log.warning(
                "sample override specified, but {SamplePath} not found in default path"
            )
        # SamplePath can be a list, so need to construct all possible paths
        sample_paths = configuration._setting_to_list(sample_paths)
        path_list = []
        for sample_path in sample_paths:
            path_list.append(general_path.replace("{SamplePath}", sample_path))
    elif sample_template_exists:
        raise ValueError(f"no path setting found for sample {sample['Name']}")
    else:
        # no need for multiple paths, and no SamplePath are present, so turn
        # the existing path into a list
        path_list = [general_path]

    # convert the contents of path_lists to paths and return them
    paths = [pathlib.Path(path) for path in path_list]
    return paths


def _variable(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> str:
    """Returns the variable the histogram will be binned in.

    For non-nominal templates, overrides the nominal variable if an alternative is
    specified for the template.

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): template considered: "Up", "Down",
            or None for nominal

    Returns:
        str: name of variable to bin histogram in
    """
    axis_variable = region["Variable"]
    # check whether a systematic is being processed
    if template is not None:
        # determine whether the template has an override specified
        axis_variable_override = utils._check_for_override(
            systematic, template, "Variable"
        )
        if axis_variable_override is not None:
            axis_variable = axis_variable_override
    return axis_variable


def _filter(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> Optional[str]:
    """Returns the filter to be applied for event selection.

    Overrides the (optional) filter provided at the region level by a sample-specific
    filter, if provided. For non-nominal templates, overrides the nominal filter if an
    alternative is specified for the template. The systematic-specific filter overrides
    the sample-specific filter if both are provided.

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): template considered: "Up", "Down",
            or None for nominal

    Returns:
        Optional[str]: expression for the filter to be used, or None for no filtering
    """
    selection_filter = region.get("Filter", None)

    # check for sample-specific overrides
    selection_filter_override = sample.get("Filter", None)
    if selection_filter_override is not None:
        selection_filter = selection_filter_override

    # check whether a systematic is being processed
    if template is not None:
        # determine whether the template has an override specified
        selection_filter_override = utils._check_for_override(
            systematic, template, "Filter"
        )
        if selection_filter_override is not None:
            selection_filter = selection_filter_override
    return selection_filter


def _weight(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> Optional[str]:
    """Returns the weight to be used for events in histograms.

    For non-nominal templates, overrides the nominal weight if an alternative is
    specified for the template.

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): template considered: "Up", "Down",
            or None for nominal

    Returns:
        Optional[str]: weight used for events when filled into histograms, or None for
        no weight
    """
    weight = sample.get("Weight", None)
    # check whether a systematic is being processed
    if template is not None:
        # determine whether the template has an override specified
        weight_override = utils._check_for_override(systematic, template, "Weight")
        if weight_override is not None:
            weight = weight_override
    return weight


def _position_in_file(
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> str:
    """Returns the location of data within a file (e.g. a tree name).

    For non-nominal templates, overrides the nominal position if an alternative is
    specified for the template.

    Args:
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): template considered: "Up", "Down",
            or None for nominal

    Returns:
        str: where in the file to find the data (the name of a tree)
    """
    position = sample["Tree"]
    # check whether a systematic is being processed
    if template is not None:
        # determine whether the template has an override specified
        position_override = utils._check_for_override(systematic, template, "Tree")
        if position_override is not None:
            position = position_override
    return position


def _binning(region: Dict[str, Any]) -> np.ndarray:
    """Returns the binning to be used in a region.

    Should eventually also support other ways of specifying bins, such as the amount of
    bins and the range to bin in.

    Args:
        region (Dict[str, Any]): containing all region information

    Raises:
        NotImplementedError: when the binning is not explicitly defined

    Returns:
        np.ndarray: bin boundaries to be used for histogram
    """
    if not region.get("Binning", False):
        raise NotImplementedError("cannot determine binning")

    return np.asarray(region["Binning"])


class _Builder:
    """Handles the instructions for backends to create histograms."""

    def __init__(
        self, histogram_folder: pathlib.Path, general_path: str, method: str
    ) -> None:
        """Creates an instance, sets histogram folder, path template and method.

        Args:
            histogram_folder (pathlib.Path): folder to save the histograms to
            general_path (str): template for paths to input files for histogram building
            method (str): backend to use for histogram production
        """
        self.histogram_folder = histogram_folder
        self.general_path = general_path
        self.method = method

    def _create_histogram(
        self,
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        template: Optional[Literal["Up", "Down"]],
    ) -> None:
        """Creates a histogram and writes it to a file.

        The histogram is created for the region-sample-systematic-template specified in
        the argument.

        Args:
            region (Dict[str, Any]): containing all region information
            sample (Dict[str, Any]): containing all sample information
            systematic (Dict[str, Any]): containing all systematic information
            template (Optional[Literal["Up", "Down"]]): template considered: "Up",
                "Down", or None for nominal

        Raises:
            NotImplementedError: when requesting an unknown backend
        """
        ntuple_paths = _ntuple_paths(
            self.general_path, region, sample, systematic, template
        )
        pos_in_file = _position_in_file(sample, systematic, template)
        variable = _variable(region, sample, systematic, template)
        bins = _binning(region)
        weight = _weight(region, sample, systematic, template)
        selection_filter = _filter(region, sample, systematic, template)

        # obtain the histogram
        if self.method == "uproot":
            from cabinetry.contrib import histogram_creator

            histogram = histogram_creator.with_uproot(
                ntuple_paths,
                pos_in_file,
                variable,
                bins,
                weight=weight,
                selection_filter=selection_filter,
            )

        else:
            raise NotImplementedError(f"unknown backend {self.method}")

        # store information in a Histogram instance and save it
        utils._name_and_save(
            self.histogram_folder,
            histo.Histogram(histogram),
            region,
            sample,
            systematic,
            template,
        )

    def _wrap_custom_template_builder(
        self, func: route.UserTemplateFunc
    ) -> route.ProcessorFunc:
        """Returns function that executes custom template builder and saves histogram.

        Wrapper for custom template builder functions that return a
        ``boost_histogram.Histogram``.

        Args:
            func (cabinetry.route.UserTemplateFunc): user-defined template builder

        Returns:
            cabinetry.route.ProcessorFunc: wrapped template builder
        """
        # decorating this with functools.wraps will keep the name of the wrapped
        # function the same, however the signature of the wrapped function is slightly
        # different (the return value becomes None)
        @functools.wraps(func)
        def wrapper(
            region: Dict[str, Any],
            sample: Dict[str, Any],
            systematic: Dict[str, Any],
            template: Optional[Literal["Up", "Down"]],
        ) -> None:
            """Executes a user-defined function that returns a histogram and saves it.

            Returns None, to turn the user-defined function into a ProcessorFunc when
            wrapped with this.

            Args:
                region (Dict[str, Any]): containing all region information
                sample (Dict[str, Any]): containing all sample information
                systematic (Dict[str, Any]): containing all systematic information
                template (Optional[Literal["Up", "Down"]]): template considered: "Up",
                    "Down", or None for nominal
            """
            histogram = func(region, sample, systematic, template)
            if not isinstance(histogram, bh.Histogram):
                raise TypeError(
                    f"{func.__name__} must return a boost_histogram.Histogram"
                )
            utils._name_and_save(
                self.histogram_folder,
                histo.Histogram(histogram),
                region,
                sample,
                systematic,
                template,
            )

        return wrapper
