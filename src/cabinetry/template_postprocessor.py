from . import histogram_wrapper


def _adjust_histogram(histogram):
    """
    create a new modified histogram
    """
    new_histogram = histogram  # for now, the histogram is unchanged
    return new_histogram


def run(config, histogram_folder):
    """
    apply post-processing as needed for all histograms
    """
    print("# applying post-processing to histograms")
    # loop over all histograms
    for sample in config["Samples"]:
        for region in config["Regions"]:
            for systematic in [{"Name": "nominal"}]:
                histogram_name = histogram_wrapper._build_histogram_name(
                    sample, region, systematic
                )
                histogram = histogram_wrapper.load_histogram(
                    histogram_folder, histogram_name
                )
                new_histogram = _adjust_histogram(histogram)
                histogram_wrapper.save_histogram(
                    new_histogram, histogram_folder, histogram_name + "_modified"
                )
