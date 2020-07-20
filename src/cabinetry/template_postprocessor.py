import copy
import logging
import numpy as np
from pathlib import Path

from . import histo


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram, name):
    """replace nan stat. unc. by zero for a histogram, modifies the
    histogram handed over in the argument

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to fix
        name (str): histogram name for logging
    """
    nan_pos = np.where(np.isnan(histogram.stdev))[0]
    if len(nan_pos) > 0:
        log.debug("fixing ill-defined stat. unc. for %s", name)
        histogram.view().variance = np.nan_to_num(histogram.stdev ** 2, nan=0.0)


def apply_postprocessing(histogram, name):
    """Create a new modified histogram, currently only calling the
    stat. uncertainty fix. The histogram in the function argument
    stays unchanged.

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to postprocess
        name (str): histogram name for logging

    Returns:
        cabinetry.histo.Histogram: the fixed histogram
    """
    # copy histogram to new object to leave it unchanged
    adjusted_histogram = copy.deepcopy(histogram)
    _fix_stat_unc(adjusted_histogram, name)
    return adjusted_histogram


def run(config, histogram_folder):
    """apply post-processing as needed for all histograms

    Args:
        config (dict): cabinetry configuration
        histogram_folder (str): folder containing the histograms
    """
    log.info("applying post-processing to histograms")
    # loop over all histograms
    for region in config["Regions"]:
        for sample in config["Samples"]:
            for systematic in [{"Name": "nominal"}]:
                # need to add histogram-based systematics here as well
                histogram = histo.Histogram.from_config(
                    histogram_folder, region, sample, systematic, modified=False
                )
                histogram_name = histo.build_name(region, sample, systematic)
                new_histogram = apply_postprocessing(histogram, histogram_name)
                histogram.validate(histogram_name)
                new_histo_path = Path(histogram_folder) / (histogram_name + "_modified")
                new_histogram.save(new_histo_path)
