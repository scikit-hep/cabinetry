import functools
import logging
import pathlib
from typing import Any, Dict, Optional

import boost_histogram as bh
import numpy as np

from . import histo
from . import route


log = logging.getLogger(__name__)


def _check_for_override(
    systematic: Dict[str, Any], template: str, option: str
) -> Optional[str]:
    """Given a systematic and a string specifying which template is currently under consideration,
    check whether the systematic defines an override for an option. Return the override if it
    exists, otherwise return None.

    Args:
        systematic (Dict[str, Any]): containing all systematic information
        template (str): template to consider: "Nominal", "Up", "Down"
        option (str): the option for which the presence of an override is checked

    Returns:
        Optional[str]: either None if no override exists, or the override
    """
    return systematic.get(template, {}).get(option, None)


def _get_ntuple_path(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: str,
) -> pathlib.Path:
    """determine the path to ntuples from which a histogram has to be built
    for non-nominal templates, override the nominal path if an alternative is
    specified for the template

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (str): which template is considered: "Nominal", "Up", "Down"

    Returns:
        pathlib.Path: path where the ntuples are located
    """
    path_str = sample["Path"]
    # check whether a systematic is being processed
    if systematic.get("Name", "nominal") != "nominal":
        # determine whether the template has an override specified
        path_str_override = _check_for_override(systematic, template, "Path")
        if path_str_override is not None:
            path_str = path_str_override
    path = pathlib.Path(path_str)
    return path


def _get_variable(region: Dict[str, Any]) -> str:
    """construct the variable the histogram will be binned in

    Args:
        region (Dict[str, Any]): containing all region information

    Returns:
        str: name of variable to bin histogram in
    """
    axis_variable = region["Variable"]
    return axis_variable


def _get_filter(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: str,
) -> Optional[str]:
    """construct the filter to be applied for event selection
    for non-nominal templates, override the nominal filter if an alternative is
    specified for the template

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (str): which template is considered: "Nominal", "Up", "Down"

    Returns:
        Optional[str]: expression for the filter to be used, or None for no filtering
    """
    selection_filter = region.get("Filter", None)
    # check whether a systematic is being processed
    if systematic.get("Name", "nominal") != "nominal":
        # determine whether the template has an override specified
        selection_filter_override = _check_for_override(systematic, template, "Filter")
        if selection_filter_override is not None:
            selection_filter = selection_filter_override
    return selection_filter


def _get_weight(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: str,
) -> Optional[str]:
    """find the weight to be used for the events in the histogram
    for non-nominal templates, override the nominal weight if an alternative is
    specified for the template

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (str): which template is considered: "Nominal", "Up", "Down"

    Returns:
        Optional[str]: weight used for events when filled into histograms, or None for no weight
    """
    weight = sample.get("Weight", None)
    # check whether a systematic is being processed
    if systematic.get("Name", "nominal") != "nominal":
        # determine whether the template has an override specified
        weight_override = _check_for_override(systematic, template, "Weight")
        if weight_override is not None:
            weight = weight_override
    return weight


def _get_position_in_file(
    sample: Dict[str, Any], systematic: Dict[str, Any], template: str
) -> str:
    """the file might have some substructure, this specifies where in the file
    the data is
    for non-nominal templates, override the nominal position if an alternative is
    specified for the template

    Args:
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (str): which template is considered: "Nominal", "Up", "Down"

    Returns:
        str: where in the file to find the data (right now the name of a tree)
    """
    position = sample["Tree"]
    # check whether a systematic is being processed
    if systematic.get("Name", "nominal") != "nominal":
        # determine whether the template has an override specified
        position_override = _check_for_override(systematic, template, "Tree")
        if position_override is not None:
            position = position_override
    return position


def _get_binning(region: Dict[str, Any]) -> np.ndarray:
    """determine the binning to be used in a given region
    should eventually also support other ways of specifying bins,
    such as the amount of bins and the range to bin in

    Args:
        region (Dict[str, Any]): containing all region information

    Raises:
        NotImplementedError: when the binning is not explicitly defined

    Returns:
        numpy.ndarray: bin boundaries to be used for histogram
    """
    if not region.get("Binning", False):
        raise NotImplementedError("cannot determine binning")

    return np.asarray(region["Binning"])


class _Builder:
    """class to handle the instructions for backends to create histograms
    """

    def __init__(self, histogram_folder: pathlib.Path, method: str) -> None:
        """create an instance, set folder and method

        Args:
            histogram_folder (pathlib.Path): folder to save the histograms to
            method (str): backend to use for histogram production
        """
        self.histogram_folder = histogram_folder
        self.method = method

    def _create_histogram(
        self,
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        template: str,
    ) -> None:
        """function to create a histogram and write it to a file for the template
        specified via the region-sample-systematic-template information

        Args:
            region (Dict[str, Any]): specifying region information
            sample (Dict[str, Any]): specifying sample information
            systematic (Dict[str, Any]): specifying systematic information
            template (str): name of the template: "Nominal", "Up", "Down"

        Raises:
            NotImplementedError: when requesting an unknown backend
        """
        ntuple_path = _get_ntuple_path(region, sample, systematic, template)
        pos_in_file = _get_position_in_file(sample, systematic, template)
        variable = _get_variable(region)
        bins = _get_binning(region)
        weight = _get_weight(region, sample, systematic, template)
        selection_filter = _get_filter(region, sample, systematic, template)

        # obtain the histogram
        if self.method == "uproot":
            from cabinetry.contrib import histogram_creation

            yields, stdev = histogram_creation.from_uproot(
                ntuple_path,
                pos_in_file,
                variable,
                bins,
                weight=weight,
                selection_filter=selection_filter,
            )

        else:
            raise NotImplementedError(f"unknown backend {self.method}")

        # store information in a Histogram instance and save it
        histogram = histo.Histogram.from_arrays(bins, yields, stdev)
        self._name_and_save(histogram, region, sample, systematic, template)

    def _name_and_save(
        self,
        histogram: histo.Histogram,
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        template: str,
    ) -> None:
        """generate a unique name for the histogram and save it

        Args:
            histogram (histo.Histogram): histogram to save
            region (Dict[str, Any]): dict with region information
            sample (Dict[str, Any]): dict with sample information
            systematic (Dict[str, Any]): dict with systematic information
            template (str): name of the template: "Nominal", "Up", "Down"
        """
        # generate a name for the histogram
        histogram_name = histo.build_name(region, sample, systematic, template)

        # check the histogram for common issues
        histogram.validate(histogram_name)

        # save it
        histo_path = self.histogram_folder / histogram_name
        histogram.save(histo_path)

    def _wrap_custom_template_builder(
        self, func: route.UserTemplateFunc,
    ) -> route.ProcessorFunc:
        """Wrapper for custom template builder functions that return a ``boost_histogram.Histogram``.
        Returns a function that executes the custom template builder and saves the resulting
        histogram.

        Args:
            func (cabinetry.route.UserTemplateFunc): user-defined template builder

        Returns:
            cabinetry.route.ProcessorFunc: wrapped template builder
        """
        # decorating this with functools.wraps will keep the name of the wrapped function the same,
        # however the signature of the wrapped function is slightly different (the return value
        # becomes None)
        @functools.wraps(func)
        def wrapper(
            region: Dict[str, Any],
            sample: Dict[str, Any],
            systematic: Dict[str, Any],
            template: str,
        ) -> None:
            """Takes a user-defined function that returns a histogram, executes that function and
            saves the histogram. Returns None, to turn the user-defined function into a
            ProcessorFunc when wrapped with this.

            Args:
                region (Dict[str, Any]): dict with region information
                sample (Dict[str, Any]): dict with sample information
                systematic (Dict[str, Any]): dict with systematic information
                template (str): name of the template: "Nominal", "Up", "Down"
            """
            histogram = func(region, sample, systematic, template)
            if not isinstance(histogram, bh.Histogram):
                raise TypeError(
                    f"{func.__name__} must return a boost_histogram.Histogram"
                )
            self._name_and_save(
                histo.Histogram(histogram), region, sample, systematic, template
            )

        return wrapper


def create_histograms(
    config: Dict[str, Any],
    method: str = "uproot",
    router: Optional[route.Router] = None,
) -> None:
    """generate all required histograms specified by the configuration file,
    calling either a default method specified via ``method``, or a custom
    user-defined override through ``router``

    Args:
        config (Dict[str, Any]): cabinetry configuration
        method (str, optional): backend to use for histogram production, defaults to "uproot"
        router (Optional[route.Router], optional): instance of cabinetry.route.Router
            that contains user-defined overrides, defaults to None
    """
    # create an instance of the class doing the template building
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    builder = _Builder(histogram_folder, method=method)

    match_func: Optional[route.MatchFunc] = None
    if router is not None:
        # specify the wrapper for user-defined functions
        router.template_builder_wrapper = builder._wrap_custom_template_builder
        # get a function that can be queried to return a user-defined template builder
        match_func = router._find_template_builder_match

    route.apply_to_all_templates(
        config, builder._create_histogram, match_func=match_func
    )
