"""Provides a histogram class based on boost-histogram."""

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import boost_histogram as bh
import numpy as np

import cabinetry
from cabinetry._typing import Literal


log = logging.getLogger(__name__)


H = TypeVar("H", bound="Histogram")


class Histogram(bh.Histogram, family=cabinetry):
    """Holds histogram information, extends boost_histogram.Histogram."""

    @classmethod
    def from_arrays(
        cls: Type[H],
        bins: Union[List[float], np.ndarray],
        yields: Union[List[float], np.ndarray],
        stdev: Union[List[float], np.ndarray],
    ) -> H:
        """Constructs a histogram from arrays of yields and uncertainties.

        The input can be lists of ints or floats, or numpy.ndarrays.

        Args:
            bins (Union[List[float], np.ndarray]): edges of histogram bins
            yields (Union[List[float], np.ndarray]): yield per histogram bin
            stdev (Union[List[float], np.ndarray]): statistical uncertainty of yield per
                bin

        Raises:
            ValueError: when amount of bins specified via bin edges and bin contents do
                not match
            ValueError: when length of yields and stdev do not match

        Returns:
            cabinetry.histo.Histogram: the histogram instance
        """
        if len(bins) != len(yields) + 1:
            raise ValueError("bin edges need one more entry than yields")
        if len(yields) != len(stdev):
            raise ValueError("yields and stdev need to have the same shape")

        out = cls(
            bh.axis.Variable(bins, underflow=False, overflow=False),
            storage=bh.storage.Weight(),
        )
        yields = np.asarray(yields)
        stdev = np.asarray(stdev)
        out[...] = np.stack([yields, stdev**2], axis=-1)
        return out

    @classmethod
    def from_path(
        cls: Type[H], histo_path: pathlib.Path, *, modified: bool = True
    ) -> H:
        """Builds a histogram from disk.

        Loads the "modified" version of the histogram by default (which received post-
        processing).

        Args:
            histo_path (pathlib.Path): where the histogram is located
            modified (bool, optional): whether to load the modified histogram (after
                post-processing), defaults to True

        Returns:
            cabinetry.histo.Histogram: the loaded histogram
        """
        if modified:
            histo_path_modified = histo_path.parent / (histo_path.name + "_modified")
            if not histo_path_modified.with_suffix(".npz").exists():
                log.warning(
                    f"the modified histogram {histo_path_modified.with_suffix('.npz')} "
                    "does not exist"
                )
                log.warning("loading the un-modified histogram instead!")
            else:
                histo_path = histo_path_modified
        histogram_npz = np.load(histo_path.with_suffix(".npz"))
        bins = histogram_npz["bins"]
        yields = histogram_npz["yields"]
        stdev = histogram_npz["stdev"]
        return cls.from_arrays(bins, yields, stdev)

    @classmethod
    def from_config(
        cls: Type[H],
        histo_folder: Union[str, pathlib.Path],
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        *,
        template: Optional[Literal["Up", "Down"]] = None,
        modified: bool = True,
    ) -> H:
        """Loads a histogram, using information specified in the configuration file.

        To find the histogram, need to provide the folder the histogram is located in
        and the relevant information from the config: region, sample, systematic,
        template.

        Args:
            histo_folder (Union[str, patlib.Path]): folder containing all histograms
            region (Dict[str, Any]): containing all region information
            sample (Dict[str, Any]): containing all sample information
            systematic (Dict[str, Any]): containing all systematic information
            template (Optional[Literal["Up", "Down"]], optional): which template to
                consider: "Up", "Down", None for the nominal case, defaults to None
            modified (bool, optional): whether to load the modified histogram (after
                post-processing), defaults to True

        Returns:
            cabinetry.histo.Histogram: the loaded histogram
        """
        # find the histogram name given config information, and then load the histogram
        histo_name = name(region, sample, systematic, template=template)
        histo_path = pathlib.Path(histo_folder) / histo_name
        return cls.from_path(histo_path, modified=modified)

    @property
    def yields(self) -> np.ndarray:
        """Returns the yields per histogram bin.

        Returns:
            np.ndarray: yields per bin
        """
        return self.values()

    @yields.setter
    def yields(self, value: np.ndarray) -> None:
        """Updates the yields per bin.

        Args:
            value (np.ndarray): yields to set
        """
        self.view().value = value  # type: ignore[union-attr]

    @property
    def stdev(self) -> np.ndarray:
        """Returns the stat. uncertainty per histogram bin.

        Returns:
            np.ndarray: stat. uncertainty per bin
        """
        return np.sqrt(self.variances())  # type: ignore[arg-type]

    @stdev.setter
    def stdev(self, value: np.ndarray) -> None:
        """Updates the variance (by specifying the standard deviation).

        Args:
            value (np.ndarray): the standard deviation
        """
        self.view().variance = value**2  # type: ignore[misc,union-attr]

    @property
    def bins(self) -> np.ndarray:
        """Returns the bin edges.

        Returns:
            np.ndarray: bin edges
        """
        return self.axes[0].edges

    def save(self, histo_path: pathlib.Path) -> None:
        """Saves a histogram to disk.

        Args:
            histo_path (pathlib.Path): where to save the histogram
        """
        log.debug(f"saving histogram to {histo_path.with_suffix('.npz')}")

        # create output directory if it does not exist yet
        if not os.path.exists(histo_path.parent):
            os.mkdir(histo_path.parent)
        np.savez(
            histo_path.with_suffix(".npz"),
            yields=self.yields,
            stdev=self.stdev,
            bins=self.bins,
        )

    def validate(self, name: str) -> None:
        """Runs consistency checks on a histogram.

        Checks for empty bins and ill-defined statistical uncertainties. Logs warnings
        if issues are founds, but does not raise exceptions.

        Args:
            name (str): name of the histogram for logging purposes
        """
        # check for empty bins
        # using np.atleast_1d to fix deprecation warning, even though the
        # input should never need it
        empty_bins = np.where(np.atleast_1d(self.yields) == 0.0)[0]
        if len(empty_bins) > 0:
            log.warning(f"{name} has empty bins: {empty_bins}")

        # check for ill-defined stat. unc.
        nan_pos = np.where(np.isnan(self.stdev))[0]
        if len(nan_pos) > 0:
            log.warning(f"{name} has bins with ill-defined stat. unc.: {nan_pos}")

        # check whether there are any bins with ill-defined stat. uncertainty
        # but non-empty yield, those deserve a closer look
        not_empty_but_nan = [b for b in nan_pos if b not in empty_bins]
        if len(not_empty_but_nan) > 0:
            log.warning(
                f"{name} has non-empty bins with ill-defined stat. unc.: "
                f"{not_empty_but_nan}"
            )

    def normalize_to_yield(self, reference_histogram: H) -> float:
        """Normalizes a histogram to match the yield of a reference.

        Returns the normalization factor used to normalize the histogram.

        Args:
            reference_histogram (cabinetry.histo.Histogram): reference histogram to
                normalize to

        Returns:
            float: the yield ratio: un-normalized yield / normalized yield
        """
        target_integrated_yield = sum(reference_histogram.yields)
        current_integrated_yield = sum(self.yields)
        normalization_ratio = current_integrated_yield / target_integrated_yield
        # scale integrated yield to match target (also scale stdev accordingly)
        self /= normalization_ratio
        return normalization_ratio


def name(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    *,
    template: Optional[Literal["Up", "Down"]] = None,
) -> str:
    """Returns a unique name for each histogram.

    If the template is not None, the systematic is required to have a name (guaranteed
    as long as it follows the config schema).

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]], optional): which template to
            consider: "Up", "Down", None for the nominal case, defaults to None

    Returns:
        str: unique name for the histogram
    """
    name = f"{region['Name']}_{sample['Name']}"
    if template is not None:
        name += f"_{systematic['Name']}_{template}"
    name = name.replace(" ", "-")
    return name
