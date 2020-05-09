"""
it might make sense to allow serialization of histograms in various
different formats, so saving and loading should go through this wrapper
"""
import logging
import os

import numpy as np


log = logging.getLogger(__name__)


def to_dict(yields, sumw2, bins):
    """
    to more conveniently move around histogram information internally, put it
    into a dictionary
    """
    histogram = {"yields": yields, "sumw2": sumw2, "bins": bins}
    return histogram


def save(histogram, path, histogram_name):
    """
    save a histogram to disk
    """
    log.info("saving %s to %s", histogram_name, path)

    # create output directory if it does not exist yet
    if not os.path.exists(path):
        os.mkdir(path)
    np.savez(
        path + histogram_name + ".npz",
        yields=histogram["yields"],
        sumw2=histogram["sumw2"],
        bins=histogram["bins"],
    )


def load(path, histogram_name):
    """
    load a histogram from disk and convert it into dictionary form
    """
    histogram_npz = np.load(path + histogram_name + ".npz")
    yields = histogram_npz["yields"]
    sumw2 = histogram_npz["sumw2"]
    bins = histogram_npz["bins"]
    return to_dict(yields, sumw2, bins)


def _build_name(sample, region, systematic):
    """
    construct a unique name for each histogram
    param sample: the sample
    """
    name = sample["Name"] + "_" + region["Name"] + "_" + systematic["Name"]
    name = name.replace(" ", "-")
    return name


def _validate_and_fix(histogram, name):
    """
    run consistency checks on a histogram, checking for empty bins
    and ill-defined statistical uncertainties, fix issues if possible
    """
    # check for empty bins
    empty_bins = np.where(histogram["yields"] == 0.0)[0]
    if len(empty_bins) > 0:
        log.warn("%s has empty bins: %s", name, empty_bins)

    # check for ill-defined stat. unc.
    nan_pos = np.where(np.isnan(histogram["sumw2"]))[0]
    if len(nan_pos) > 0:
        log.warn("%s has bins with ill-defined stat. unc.: %s", name, nan_pos)

    # check whether there are any bins with ill-defined stat. uncertainty
    # but non-empty yield, those deserve a closer look
    not_empty_but_nan = [b for b in nan_pos if b not in empty_bins]
    if len(not_empty_but_nan) > 0:
        log.warn(
            "%s has non-empty bins with ill-defined stat. unc.: %s",
            name,
            not_empty_but_nan,
        )

    # fix ill defined stat. unc. (replace nan by 0)
    log.debug("fixing ill-defined stat. unc. for %s", name)
    histogram["sumw2"] = np.nan_to_num(histogram["sumw2"], nan=0.0)
    return histogram
