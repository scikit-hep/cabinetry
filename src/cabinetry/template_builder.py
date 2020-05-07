from . import histogram_wrapper


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
    returns either amount of bins and range,
    or the bins themselves and None
    """
    if "Binning" in region.keys():
        return region["Binning"], None
    else:
        raise NotImplementedError


def create_histograms(config, output_path, method="uproot", only_nominal=False):
    """
    generate all required histograms specified by a configuration file
    """
    print("# creating histograms")

    for sample in config["Samples"]:
        # print("- reading sample", sample["Name"])

        for region in config["Regions"]:
            # print("- in region", region["Name"])

            for isyst, systematic in enumerate(
                ([{"Name": "nominal"}] + config["Systematics"])
            ):
                # first do the nominal case, then systematics
                # optionally skip non-nominal histograms
                if isyst != 0 and only_nominal:
                    continue
                elif isyst > 0:
                    raise NotImplementedError("systematics not yet implemented")

                # print("- variation", systematic["Name"])
                ntuple_path = _get_ntuple_path(sample, region, systematic)
                pos_in_file = _get_position_in_file(sample)
                selection = _get_selection(sample, region, systematic)
                weight = _get_weight(sample, region, systematic)
                bins, bin_range = _get_binning(region)

                # obtain the histogram
                if method == "uproot":
                    from cabinetry.contrib import histogram_creation

                    histogram = histogram_creation.from_uproot(
                        ntuple_path, pos_in_file, selection, weight, bins, bin_range
                    )

                elif "method" == "ServiceX":
                    raise NotImplementedError

                else:
                    raise NotImplementedError

                # save the histogram
                histogram_name = histogram_wrapper._build_histogram_name(
                    sample, region, systematic
                )
                histogram_wrapper.save_histogram(histogram, output_path, histogram_name)
