"""
it might make sense to allow serialization of histograms in various
different formats, so saving and loading should go through this wrapper
"""
import logging
import os
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)


def to_dict(yields, sumw2, bins):
    """to more conveniently move around histogram information internally, put it
    into a dictionary

    Args:
        yields (numpy.ndarray): yield per histogram bin
        sumw2 (numpy.ndarray): statistical uncertainty of yield per bin
        bins (numpy.ndarray): edges of histogram bins

    Returns:
        dict: dictionary containing histogram information
    """
    histogram = {"yields": yields, "sumw2": sumw2, "bins": bins}
    return histogram


def save(histogram, histo_path):
    """save a histogram to disk

    Args:
        histogram (dict): the histogram to be saved
        histo_path (pathlib.Path): where to save the histogram
    """
    log.debug("saving histogram to %s", histo_path.with_suffix(".npz"))

    # create output directory if it does not exist yet
    if not os.path.exists(histo_path.parent):
        os.mkdir(histo_path.parent)
    np.savez(
        histo_path.with_suffix(".npz"),
        yields=histogram["yields"],
        sumw2=histogram["sumw2"],
        bins=histogram["bins"],
    )


def _load(histo_path, modified=True):
    """load a histogram from disk and convert it into dictionary form
    try to load the "modified" version of the histogram by default
    (which received post-processing)

    Args:
        histo_path (pathlib.Path): where the histogram is located
        modified (bool, optional): whether to load the modified histogram (after post-processing), defaults to True

    Returns:
        dict: the loaded histogram
    """
    if modified:
        histo_path_modified = histo_path.parent / (histo_path.name + "_modified")
        if not histo_path_modified.with_suffix(".npz").exists():
            log.warning(
                "the modified histogram %s does not exist",
                histo_path_modified.with_suffix(".npz"),
            )
            log.warning("loading the un-modified histogram instead!")
        else:
            histo_path = histo_path_modified
    histogram_npz = np.load(histo_path.with_suffix(".npz"))
    yields = histogram_npz["yields"]
    sumw2 = histogram_npz["sumw2"]
    bins = histogram_npz["bins"]
    return to_dict(yields, sumw2, bins)


def load_from_config(histo_folder, sample, region, systematic, modified=True):
    """load a histogram, given a folder the histograms are located in and the
    relevant information from the config: sample, region, systematic

    Args:
        histo_folder (str): folder containing all histograms
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information
        modified (bool, optional): whether to load the modified histogram (after post-processing), defaults to True

    Returns:
        tuple: a tuple containing
            dict: the loaded histogram
            str: name of the histogram
    """
    # find the histogram name given config information, and then load the histogram
    histo_name = build_name(sample, region, systematic)
    histo_path = Path(histo_folder) / histo_name
    histogram = _load(histo_path, modified)
    return histogram, histo_name


def build_name(sample, region, systematic):
    """construct a unique name for each histogram

    Args:
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information

    Returns:
        str: unique name for the histogram
    """
    name = sample["Name"] + "_" + region["Name"] + "_" + systematic["Name"]
    name = name.replace(" ", "-")
    return name


def validate(histogram, name):
    """run consistency checks on a histogram, checking for empty bins
    and ill-defined statistical uncertainties

    Args:
        histogram (dict): the histogram to validate
        name (str): name of the histogram for logging purposes
    """
    # check for empty bins
    # using np.atleast_1d to fix deprecation warning, even though the
    # input should never need it
    empty_bins = np.where(np.atleast_1d(histogram["yields"]) == 0.0)[0]
    if len(empty_bins) > 0:
        log.warning("%s has empty bins: %s", name, empty_bins)

    # check for ill-defined stat. unc.
    nan_pos = np.where(np.isnan(histogram["sumw2"]))[0]
    if len(nan_pos) > 0:
        log.warning("%s has bins with ill-defined stat. unc.: %s", name, nan_pos)

    # check whether there are any bins with ill-defined stat. uncertainty
    # but non-empty yield, those deserve a closer look
    not_empty_but_nan = [b for b in nan_pos if b not in empty_bins]
    if len(not_empty_but_nan) > 0:
        log.warning(
            "%s has non-empty bins with ill-defined stat. unc.: %s",
            name,
            not_empty_but_nan,
        )


def normalize_to_yield(histogram, reference_histogram):
    """normalize a histogram to match the yield of a reference, and return the
    modified histogram along with the normalization factor

    Args:
        histogram (dict): the histogram to be normalized
        reference_histogram (dict): reference histogram to normalize to

    Returns:
        tuple: a tuple containing
            dict: the normalized histogram
            np.float64: the yield ratio: un-normalized yield / normalized yield
    """
    target_integrated_yield = sum(reference_histogram["yields"])
    current_integrated_yield = sum(histogram["yields"])
    normalization_ratio = current_integrated_yield / target_integrated_yield
    # update integrated yield to match target
    histogram["yields"] /= normalization_ratio
    return histogram, normalization_ratio
