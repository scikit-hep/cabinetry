"""Implements histogram smoothing algorithms."""

import logging
import statistics
from typing import List, TypeVar, Union

import numpy as np


log = logging.getLogger(__name__)


# typeguard raises errors in tests when using List[float] instead of list
T = TypeVar("T", list, np.ndarray)


def _medians_353(zz: Union[List[float], np.ndarray], nbins: int) -> None:
    """Applies running median smoothing with window sizes 3, 5, 3 to input.

    Args:
        zz (Union[List[float], np.ndarray]): array to smooth
        nbins (int): number of bins in array
    """
    for i_median in range(3):
        yy = zz.copy()  # yy stays constant at each step in the loop
        medianType = 3 if i_median != 1 else 5
        ifirst = 1 if i_median != 1 else 2
        ilast = nbins - 1 if i_median != 1 else nbins - 2
        # do central elements first in (ifirst, ilast) range
        for ii in range(ifirst, ilast):
            zz[ii] = statistics.median(yy[ii - ifirst : ii - ifirst + medianType])

        if i_median == 0:  # first median 3
            # first bin, proceedings use y_1 (=yy[0]), while ROOT uses zz[0]
            zz[0] = statistics.median([3 * zz[1] - 2 * zz[2], yy[0], zz[1]])
            # last bin, proceedings use y_n (=yy[-1]), while ROOT uses zz[-1]
            zz[-1] = statistics.median([zz[-2], yy[-1], 3 * zz[-2] - 2 * zz[-3]])

        if i_median == 1:  # median 5
            zz[1] = statistics.median(yy[0:3])
            # second to last bin
            zz[-2] = statistics.median(yy[-3:])


def smooth_353qh_twice(hist: T) -> T:
    """Runs the 353QH algorithm twice and returns smooth version of the input.

    For documentation see these proceedings https://cds.cern.ch/record/186223/ on page
    292. The algorithm runs twice to avoid over-smoothing peaks and valleys. The
    algorithm is not aware of statistical uncertainties per entry in the array. See also
    https://root.cern.ch/doc/master/classTH1.html#aeb935cae10dbf9cd484bee1b6a549f83 for
    the ROOT implementation.

    Args:
        hist (Union[list, np.ndarray]): array to smooth

    Returns:
        Union[list, np.ndarray]: smooth version of input
    """
    nbins = len(hist)
    if nbins < 3:
        log.warning("at least three points needed for smoothing, no smoothing applied")
        return hist

    if isinstance(hist, np.ndarray):
        # ensure dtype is not int to avoid rounding in smooth histogram
        hist = hist.astype("float")

    zz = np.array(hist, dtype=float, copy=True)

    for i_353QH in range(2):  # run 353QH twice
        # do running median with window sizes 3, 5, 3
        _medians_353(zz, nbins)

        yy = zz.copy()

        # quadratic interpolation for flat segments
        for ii in range(2, nbins - 2):
            if zz[ii - 1] != zz[ii] or zz[ii] != zz[ii + 1]:
                continue  # not flat
            left_larger = zz[ii - 2] - zz[ii]  # two bins left larger by this amount
            right_larger = zz[ii + 2] - zz[ii]  # two bins right larger by this amount
            if left_larger * right_larger <= 0:
                continue  # current position is part of larger scale slope
            jk = 1
            if abs(right_larger) > abs(left_larger):
                jk = -1  # left is more flat than right
            yy[ii] = -0.5 * zz[ii - 2 * jk] + zz[ii] / 0.75 + zz[ii + 2 * jk] / 6
            yy[ii + jk] = 0.5 * (zz[ii + 2 * jk] - zz[ii - 2 * jk]) + zz[ii]

        # running means
        for ii in range(1, nbins - 1):
            zz[ii] = 0.25 * yy[ii - 1] + 0.5 * yy[ii] + 0.25 * yy[ii + 1]
        zz[0] = yy[0]
        zz[-1] = yy[-1]

        if i_353QH == 0:
            # algorithm has been run once
            rr = zz.copy()  # save computed values
            # calculate residuals: (original) - (after 353QH)
            zz = np.asarray([hist[ii] - zz[ii] for ii in range(0, nbins)])
            # zz is now "rough", while rr is "smooth"

        # "twicing": run 353QH again on "rough" zz and add to "smooth" rr

    hist_new = hist.copy()  # histogram after smoothing
    for ii in range(nbins):
        if min(hist) < 0:
            hist_new[ii] = rr[ii] + zz[ii]
        # result is positive if no negative values are in input
        else:
            hist_new[ii] = max(rr[ii] + zz[ii], 0)
    return hist_new
