import logging
import os
from pathlib import Path

import boost_histogram as bh
import numpy as np


log = logging.getLogger(__name__)


class Histogram(bh.Histogram):
    """class to hold histogram information, extends functionality provided
    by boost_histogram.Histogram
    """

    @classmethod
    def from_arrays(cls, bins, yields, sumw2):
        """construct a histogram from arrays of yields and uncertainties, the input
        can be lists or numpy.ndarrays

        Args:
            bins (Union[list, numpy.ndarray]): edges of histogram bins
            yields (Union[list, numpy.ndarray]): yield per histogram bin
            sumw2 (Union[list, numpy.ndarray]): statistical uncertainty of yield per bin

        Raises:
            ValueError: when amount of bins specified via bin edges and bin contents do not match
            ValueError: when length of yields and sumw2 do not match

        Returns:
            cabinetry.histo.Histogram: the histogram instance
        """
        if len(bins) != len(yields) + 1:
            raise ValueError("bin edges need one more entry than yields")
        if len(yields) != len(sumw2):
            raise ValueError("yields and sumw2 need to have the same shape")

        out = cls(
            bh.axis.Variable(bins, underflow=False, overflow=False),
            storage=bh.storage.Weight(),
        )
        yields = np.asarray(yields)
        sumw2 = np.asarray(sumw2)
        out[...] = np.stack([yields, sumw2 ** 2], axis=-1)
        return out

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
        bins = histogram_npz["bins"]
        yields = histogram_npz["yields"]
        sumw2 = histogram_npz["sumw2"]
        return cls.from_arrays(bins, yields, sumw2)

    @classmethod
    def from_config(cls, histo_folder, region, sample, systematic, modified=True):
        """load a histogram, given a folder the histogram is located in and the
        relevant information from the config: region, sample, systematic

        Args:
            histo_folder (str): folder containing all histograms
            region (dict): containing all region information
            sample (dict): containing all sample information
            systematic (dict): containing all systematic information
            modified (bool, optional): whether to load the modified histogram (after post-processing), defaults to True

        Returns:
            cabinetry.histo.Histogram: the loaded histogram
        """
        # find the histogram name given config information, and then load the histogram
        histo_name = build_name(region, sample, systematic)
        histo_path = Path(histo_folder) / histo_name
        return cls.from_path(histo_path, modified)

    @property
    def yields(self):
        return self.view().value

    @property
    def sumw2(self):
        return np.sqrt(self.view().variance)

    @property
    def bins(self):
        return self.axes[0].edges

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
        self.view().value /= normalization_ratio
        return normalization_ratio


def build_name(region, sample, systematic):
    """construct a unique name for each histogram

    Args:
        region (dict): containing all region information
        sample (dict): containing all sample information
        systematic (dict): containing all systematic information

    Returns:
        str: unique name for the histogram
    """
    name = region["Name"] + "_" + sample["Name"] + "_" + systematic["Name"]
    name = name.replace(" ", "-")
    return name
