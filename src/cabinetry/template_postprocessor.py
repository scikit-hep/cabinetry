import logging
import numpy as np

from . import histo


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram, name):
    """
    replace nan stat. unc. by zero
    """
    nan_pos = np.where(np.isnan(histogram["sumw2"]))[0]
    if len(nan_pos) > 0:
        log.debug("fixing ill-defined stat. unc. for %s", name)
        histogram["sumw2"] = np.nan_to_num(histogram["sumw2"], nan=0.0)
    return histogram


def adjust_histogram(histogram, name):
    """
    create a new modified histogram
    """
    new_histogram = _fix_stat_unc(histogram, name)
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
                histogram_name = histo.build_name(sample, region, systematic)
                histogram = histo.load(histogram_folder, histogram_name, modified=False)
                new_histogram = adjust_histogram(histogram, histogram_name)
                histo.validate(histogram, histogram_name)
                histo.save(
                    new_histogram, histogram_folder, histogram_name + "_modified"
                )
