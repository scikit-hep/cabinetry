import logging
from pathlib import Path
from typing import Optional

import numpy as np

from . import configuration
from . import histo


log = logging.getLogger(__name__)


def _check_for_override(systematic: dict, template: str, option: str) -> Optional[str]:
    """Given a systematic and a string specifying which template is currently under consideration,
    check whether the systematic defines an override for an option. Return the override if it
    exists, otherwise return None.

    Args:
        systematic (dict): containing all systematic information
        template (str): template to consider: "Nominal", "Up", "Down"
        option (str): the option for which the presence of an override is checked

    Returns:
        Optional[str]: either None if no override exists, or the override
    """
    return systematic.get(template, {}).get(option, None)


def _get_ntuple_path(
    region: dict, sample: dict, systematic: dict, template: str
) -> Path:
    """determine the path to ntuples from which a histogram has to be built
    for non-nominal templates, override the nominal path if an alternative is
    specified for the template

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information
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
    path = Path(path_str)
    return path


def _get_variable(region: dict) -> str:
    """construct the variable the histogram will be binned in

    Args:
        region (dict): containing all region information

    Returns:
        str: name of variable to bin histogram in
    """
    axis_variable = region["Variable"]
    return axis_variable


def _get_filter(
    region: dict, sample: dict, systematic: dict, template: str
) -> Optional[str]:
    """construct the filter to be applied for event selection
    for non-nominal templates, override the nominal filter if an alternative is
    specified for the template

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information
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


def _get_weight(region: dict, sample: dict, systematic: dict, template: str) -> str:
    """find the weight to be used for the events in the histogram
    for non-nominal templates, override the nominal weight if an alternative is
    specified for the template

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information
        template (str): which template is considered: "Nominal", "Up", "Down"

    Returns:
        str: weight used for events when filled into histograms
    """
    weight = sample.get("Weight", None)
    # check whether a systematic is being processed
    if systematic.get("Name", "nominal") != "nominal":
        # determine whether the template has an override specified
        weight_override = _check_for_override(systematic, template, "Weight")
        if weight_override is not None:
            weight = weight_override
    return weight


def _get_position_in_file(sample: dict, systematic: dict, template: str) -> str:
    """the file might have some substructure, this specifies where in the file
    the data is
    for non-nominal templates, override the nominal position if an alternative is
    specified for the template

    Args:
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information
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


def _get_binning(region: dict) -> np.ndarray:
    """determine the binning to be used in a given region
    should eventually also support other ways of specifying bins,
    such as the amount of bins and the range to bin in

    Args:
        region (dict): containing all region information

    Raises:
        NotImplementedError: when the binning is not explicitly defined

    Returns:
        numpy.ndarray: bin boundaries to be used for histogram
    """
    if not region.get("Binning", False):
        raise NotImplementedError("cannot determine binning")

    return np.asarray(region["Binning"])


def create_histograms(
    config: dict, folder_path_str: str, method: str = "uproot"
) -> None:
    """generate all required histograms specified by a configuration file
    a tool providing histograms should provide bin yields and statistical
    uncertainties, as well as the bin edges

    Args:
        config (dict): cabinetry configuration
        folder_path_str (str): folder to save the histograms to
        method (str, optional): backend to use for histogram production, defaults to "uproot"

    Raises:
        NotImplementedError: when requesting the ServiceX backend
        NotImplementedError: when requesting another unknown backend
    """
    log.info("creating histograms")

    for region in config["Regions"]:
        log.debug(f"  in region {region['Name']}")

        for sample in config["Samples"]:
            log.debug(f"  reading sample {sample['Name']}")

            for systematic in [{"Name": "nominal"}] + config["Systematics"]:

                log.debug(f"  variation {systematic['Name']}")

                # determine how many templates need to be considered
                if systematic["Name"] == "nominal":
                    # only nominal template is needed
                    templates = ["Nominal"]
                else:
                    # systematics can have up and down template
                    templates = ["Up", "Down"]

                for template in templates:

                    # determine whether a histogram is needed for this
                    # specific combination of sample-region-systematic-template
                    histo_needed = configuration.histogram_is_needed(
                        region, sample, systematic, template
                    )

                    if not histo_needed:
                        # no further action is needed, continue with the next region-sample-systematic combination
                        continue

                    ntuple_path = _get_ntuple_path(region, sample, systematic, template)
                    pos_in_file = _get_position_in_file(sample, systematic, template)
                    variable = _get_variable(region)
                    selection_filter = _get_filter(region, sample, systematic, template)
                    weight = _get_weight(region, sample, systematic, template)
                    bins = _get_binning(region)

                    # obtain the histogram
                    if method == "uproot":
                        from cabinetry.contrib import histogram_creation

                        yields, stdev = histogram_creation.from_uproot(
                            ntuple_path,
                            pos_in_file,
                            variable,
                            bins,
                            weight,
                            selection_filter,
                        )

                    elif method == "ServiceX":
                        raise NotImplementedError("ServiceX not yet implemented")

                    else:
                        raise NotImplementedError("unknown backend")

                    # store information in a Histogram instance
                    histogram = histo.Histogram.from_arrays(bins, yields, stdev)

                    # generate a name for the histogram
                    histogram_name = histo.build_name(
                        region, sample, systematic, template
                    )

                    # check the histogram for common issues
                    histogram.validate(histogram_name)

                    # save it
                    histo_path = Path(folder_path_str) / histogram_name
                    histogram.save(histo_path)
