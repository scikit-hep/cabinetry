import logging
import numpy as np
from pathlib import Path

from . import histo


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram, name):
    """replace nan stat. unc. by zero

    Args:
        histogram (dict): the histogram to fix
        name (str): histogram name for logging

    Returns:
        dict: the fixed histogram
    """
    nan_pos = np.where(np.isnan(histogram["sumw2"]))[0]
    if len(nan_pos) > 0:
        log.debug("fixing ill-defined stat. unc. for %s", name)
        histogram["sumw2"] = np.nan_to_num(histogram["sumw2"], nan=0.0)
    return histogram


def adjust_histogram(histogram, name):
    """create a new modified histogram, currently only calling the
    stat. uncertainty fix

    Args:
        histogram (dict): the histogram to modify
        name (str): histogram name for logging

    Returns:
        dict: the modified histogram
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
                histogram, histogram_name = histo.load_from_config(
                    histogram_folder, sample, region, systematic, modified=False
                )
                new_histogram = adjust_histogram(histogram, histogram_name)
                histo.validate(histogram, histogram_name)
                new_histo_path = Path(histogram_folder) / (histogram_name + "_modified")
                histo.save(new_histogram, new_histo_path)
