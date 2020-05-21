import logging
from pathlib import Path

from . import histo


log = logging.getLogger(__name__)


def _get_ntuple_path(sample, region, systematic):
    """determine the path to ntuples from which a histogram has to be built

    Args:
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information

    Returns:
        pathlib.Path: path where the ntuples are located
    """
    path_str = sample["Path"]
    path = Path(path_str)
    return path


def _get_variable(sample, region, systematic):
    """construct the variable the histogram will be binned in

    Args:
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information

    Returns:
        str: name of variable to bin histogram in
    """
    axis_variable = region["Variable"]
    return axis_variable


def _get_filter(sample, region, systematic):
    """construct the filter to be applied for event selection

    Args:
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information

    Returns:
        str: expression for the filter to be used, or None for no filtering
    """
    selection_filter = region.get("Filter", None)
    return selection_filter


def _get_weight(sample, region, systematic):
    """find the weight to be used for the events in the histogram

    Args:
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information

    Returns:
        str: weight used for events when filled into histograms
    """
    weight = sample.get("Weight", None)
    return weight


def _get_position_in_file(sample):
    """the file might have some substructure, this specifies where in the file
    the data is

    Args:
        sample (dict): containing all sample information

    Returns:
        str: where in the file to find the data, (right now the name of a tree)
    """
    return sample["Tree"]


def _get_binning(region):
    """determine the binning to be used in a given region
    should eventually also support other ways of specifying bins,
    such as the amount of bins and the range to bin in

    Args:
        region (dict): containing all region information

    Raises:
        NotImplementedError: when the binning is not explicitly defined

    Returns:
        list: bin boundaries to be used for histogram
    """
    if region.get("Binning", False):
        return region["Binning"]
    else:
        raise NotImplementedError


def create_histograms(config, folder_path_str, method="uproot", only_nominal=False):
    """generate all required histograms specified by a configuration file
    a tool providing histograms should provide bin yields and statistical
    uncertainties, as well as the bin edges

    Args:
        config (dict): the cabinetry config
        folder_path_str (str): folder to save the histograms to
        method (str, optional): backend to use for histogram production, defaults to "uproot"
        only_nominal (bool, optional): whether to only produce nominal histograms, defaults to False

    Raises:
        NotImplementedError: when systematic variations based on histograms are requested (not supported yet)
        NotImplementedError: when requesting the ServiceX backend
        NotImplementedError: when requesting another unknown backend
    """
    log.info("creating histograms")

    for sample in config["Samples"]:
        log.debug("  reading sample %s", sample["Name"])

        for region in config["Regions"]:
            log.debug("  in region %s", region["Name"])

            for isyst, systematic in enumerate(
                ([{"Name": "nominal"}] + config["Systematics"])
            ):
                # first do the nominal case, then systematics
                # optionally skip non-nominal histograms
                if isyst != 0 and only_nominal:
                    continue
                elif isyst > 0:
                    raise NotImplementedError("systematics not yet implemented")

                log.debug("  variation %s", systematic["Name"])
                ntuple_path = _get_ntuple_path(sample, region, systematic)
                pos_in_file = _get_position_in_file(sample)
                variable = _get_variable(sample, region, systematic)
                selection_filter = _get_filter(sample, region, systematic)
                weight = _get_weight(sample, region, systematic)
                bins = _get_binning(region)

                # obtain the histogram
                if method == "uproot":
                    from cabinetry.contrib import histogram_creation

                    yields, sumw2 = histogram_creation.from_uproot(
                        ntuple_path,
                        pos_in_file,
                        variable,
                        bins,
                        weight,
                        selection_filter,
                    )

                elif "method" == "ServiceX":
                    raise NotImplementedError

                else:
                    raise NotImplementedError

                # convert the information into a dictionary for easier handling
                histogram = histo.to_dict(yields, sumw2, bins)

                # generate a name for the histogram
                histogram_name = histo.build_name(sample, region, systematic)

                # check the histogram for common issues
                histo.validate(histogram, histogram_name)

                # save it
                histo_path = Path(folder_path_str) / histogram_name
                histo.save(histogram, histo_path)
