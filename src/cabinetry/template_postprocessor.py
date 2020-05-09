import logging

from . import histogram_wrapper


log = logging.getLogger(__name__)


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
    log.info("applying post-processing to histograms")
    # loop over all histograms
    for sample in config["Samples"]:
        for region in config["Regions"]:
            for systematic in [{"Name": "nominal"}]:
                histogram_name = histogram_wrapper._build_name(
                    sample, region, systematic
                )
                histogram = histogram_wrapper.load(histogram_folder, histogram_name)
                new_histogram = _adjust_histogram(histogram)
                histogram_wrapper.save(
                    new_histogram, histogram_folder, histogram_name + "_modified"
                )
