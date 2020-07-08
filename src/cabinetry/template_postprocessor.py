import logging
import numpy as np
from pathlib import Path

from . import histo


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram, name):
    """replace nan stat. unc. by zero

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to fix
        name (str): histogram name for logging

    Returns:
        cabinetry.histo.Histogram: the fixed histogram
    """
    nan_pos = np.where(np.isnan(histogram.sumw2))[0]
    if len(nan_pos) > 0:
        log.debug("fixing ill-defined stat. unc. for %s", name)
        histogram.sumw2 = np.nan_to_num(histogram.sumw2, nan=0.0)
    return histogram


def adjust_histogram(histogram, name):
    """create a new modified histogram, currently only calling the
    stat. uncertainty fix

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to modify
        name (str): histogram name for logging

    Returns:
        cabinetry.histo.Histogram: the modified histogram
    """
    new_histogram = _fix_stat_unc(histogram, name)
    return new_histogram


def run(config, histogram_folder):
    """apply post-processing as needed for all histograms

    Args:
        config (dict): cabinetry configuration
        histogram_folder (str): folder containing the histograms
    """
    log.info("applying post-processing to histograms")
    # loop over all histograms
    for sample in config["Samples"]:
        for region in config["Regions"]:
            for systematic in [{"Name": "nominal"}]:
                # need to add histogram-based systematics here as well
                histogram = histo.Histogram.from_config(
                    histogram_folder, sample, region, systematic, modified=False
                )
                histogram_name = histo.build_name(sample, region, systematic)
                new_histogram = adjust_histogram(histogram, histogram_name)
                histogram.validate(histogram_name)
                new_histo_path = Path(histogram_folder) / (histogram_name + "_modified")
                new_histogram.save(new_histo_path)
