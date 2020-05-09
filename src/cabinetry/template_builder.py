import logging

from . import histo


log = logging.getLogger(__name__)


def _get_ntuple_path(sample, region, systematic):
    """
    determine the path to ntuples from which a histogram has to be built
    """
    path = sample["Path"]
    return path


def _get_selection(sample, region, systematic):
    """
    construct the selection to be executed to obtain the histogram
    """
    axis_variable = region["Variable"]
    return axis_variable


def _get_weight(sample, region, systematic):
    """
    find the weight to be used for the events in the histogram
    """
    weight = sample.get("Weight", None)
    return weight


def _get_position_in_file(sample):
    """
    the file might have some substructure, this specifies where in the file
    the data is
    """
    return sample["Tree"]


def _get_binning(region):
    """
    determine the binning to be used in a given region
    should eventually also support other ways of specifying bins,
    such as the amount of bins and the range to bin in
    """
    if "Binning" in region.keys():
        return region["Binning"]
    else:
        raise NotImplementedError


def create_histograms(config, output_path, method="uproot", only_nominal=False):
    """
    generate all required histograms specified by a configuration file
    a tool providing histograms should provide bin yields and statistical
    uncertainties, as well as the bin edges
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
                selection = _get_selection(sample, region, systematic)
                weight = _get_weight(sample, region, systematic)
                bins = _get_binning(region)

                # obtain the histogram
                if method == "uproot":
                    from cabinetry.contrib import histogram_creation

                    yields, sumw2 = histogram_creation.from_uproot(
                        ntuple_path, pos_in_file, selection, weight, bins
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
                histo.save(histogram, output_path, histogram_name)
