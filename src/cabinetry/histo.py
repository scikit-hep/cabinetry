"""
it might make sense to allow serialization of histograms in various
different formats, so saving and loading should go through this wrapper
"""
import logging
import os
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)


class Histogram:
    """class to hold histogram information
    """

    def __init__(self, yields, sumw2, bins):
        """constructor building histogram from arrays

        Args:
            yields (numpy.ndarray): yield per histogram bin
            sumw2 (numpy.ndarray): statistical uncertainty of yield per bin
            bins (numpy.ndarray): edges of histogram bins
        """
        self.yields = yields
        self.sumw2 = sumw2
        self.bins = bins

    @classmethod
    def from_path(cls, histo_path, modified=True):
        """build a histogram from disk
        try to load the "modified" version of the histogram by default
        (which received post-processing)

        Args:
            histo_path (pathlib.Path): where the histogram is located
            modified (bool, optional): whether to load the modified histogram (after post-processing), defaults to True

        Returns:
            cabinetry.histo.Histogram: the loaded histogram
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
        return cls(yields, sumw2, bins)

    @classmethod
    def from_config(cls, histo_folder, sample, region, systematic, modified=True):
        """load a histogram, given a folder the histogram is located in and the
        relevant information from the config: sample, region, systematic

        Args:
            histo_folder (str): folder containing all histograms
            sample (dict): containing all sample information
            region (dict): containing all region information
            systematic (dict): containing all systematic information
            modified (bool, optional): whether to load the modified histogram (after post-processing), defaults to True

        Returns:
            cabinetry.histo.Histogram: the loaded histogram
        """
        # find the histogram name given config information, and then load the histogram
        histo_name = build_name(sample, region, systematic)
        histo_path = Path(histo_folder) / histo_name
        return cls.from_path(histo_path, modified)

    def save(self, histo_path):
        """save a histogram to disk

        Args:
            histo_path (pathlib.Path): where to save the histogram
        """
        log.debug("saving histogram to %s", histo_path.with_suffix(".npz"))

        # create output directory if it does not exist yet
        if not os.path.exists(histo_path.parent):
            os.mkdir(histo_path.parent)
        np.savez(
            histo_path.with_suffix(".npz"),
            yields=self.yields,
            sumw2=self.sumw2,
            bins=self.bins,
        )

    def validate(self, name):
        """run consistency checks on a histogram, checking for empty bins
        and ill-defined statistical uncertainties

        Args:
            name (str): name of the histogram for logging purposes
        """
        # check for empty bins
        # using np.atleast_1d to fix deprecation warning, even though the
        # input should never need it
        empty_bins = np.where(np.atleast_1d(self.yields) == 0.0)[0]
        if len(empty_bins) > 0:
            log.warning("%s has empty bins: %s", name, empty_bins)

        # check for ill-defined stat. unc.
        nan_pos = np.where(np.isnan(self.sumw2))[0]
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

    def normalize_to_yield(self, reference_histogram):
        """normalize a histogram to match the yield of a reference, and return the
        normalization factor

        Args:
            reference_histogram (cabinetry.histo.Histogram): reference histogram to normalize to

        Returns:
            np.float64: the yield ratio: un-normalized yield / normalized yield
        """
        target_integrated_yield = sum(reference_histogram.yields)
        current_integrated_yield = sum(self.yields)
        normalization_ratio = current_integrated_yield / target_integrated_yield
        # update integrated yield to match target
        self.yields /= normalization_ratio
        return normalization_ratio


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
