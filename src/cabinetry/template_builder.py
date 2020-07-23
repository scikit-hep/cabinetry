import logging
from pathlib import Path

import numpy as np

from . import configuration
from . import histo


log = logging.getLogger(__name__)


def _get_ntuple_path(region, sample, systematic):
    """determine the path to ntuples from which a histogram has to be built

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information

    Raises:
        NotImplementedError: if ntuple path treatment is not yet implemented

    Returns:
        pathlib.Path: path where the ntuples are located
    """
    if systematic["Name"] == "nominal":
        path_str = sample["Path"]
    elif systematic["Type"] == "NormPlusShape":
        path_str = systematic["Up"]["Path"]
    else:
        raise NotImplementedError("ntuple path treatment not yet defined")
    path = Path(path_str)
    return path


def _get_variable(region, sample, systematic):
    """construct the variable the histogram will be binned in

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information

    Returns:
        str: name of variable to bin histogram in
    """
    axis_variable = region["Variable"]
    return axis_variable


def _get_filter(region, sample, systematic):
    """construct the filter to be applied for event selection

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information

    Returns:
        str: expression for the filter to be used, or None for no filtering
    """
    selection_filter = region.get("Filter", None)
    return selection_filter


def _get_weight(region, sample, systematic):
    """find the weight to be used for the events in the histogram

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information

    Returns:
        str: weight used for events when filled into histograms
    """
    weight = sample.get("Weight", None)
    return weight


def _get_position_in_file(sample, systematic):
    """the file might have some substructure, this specifies where in the file
    the data is

    Args:
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information

    Raises:
        NotImplementedError: if finding the position in file is not yet implemented

    Returns:
        str: where in the file to find the data, (right now the name of a tree)
    """
    if systematic["Name"] == "nominal":
        position = sample["Tree"]
    elif systematic["Type"] == "NormPlusShape":
        position = systematic["Up"]["Tree"]
    else:
        raise NotImplementedError("ntuple path treatment not yet defined")
    return position


def _get_binning(region):
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


def create_histograms(config, folder_path_str, method="uproot"):
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
                # determine whether a histogram is needed for this
                # specific combination of sample-region-systematic
                histo_needed = configuration.histogram_is_needed(
                    region, sample, systematic
                )

                if not histo_needed:
                    # no further action is needed, continue with the next region-sample-systematic combination
                    continue

                log.debug(f"  variation {systematic['Name']}")
                ntuple_path = _get_ntuple_path(region, sample, systematic)
                pos_in_file = _get_position_in_file(sample, systematic)
                variable = _get_variable(region, sample, systematic)
                selection_filter = _get_filter(region, sample, systematic)
                weight = _get_weight(region, sample, systematic)
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
                histogram_name = histo.build_name(region, sample, systematic)

                # check the histogram for common issues
                histogram.validate(histogram_name)

                # save it
                histo_path = Path(folder_path_str) / histogram_name
                histogram.save(histo_path)
