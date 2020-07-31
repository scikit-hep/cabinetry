import pathlib
from typing import Optional, Tuple

import boost_histogram as bh
import numpy as np
import uproot4 as uproot


def from_uproot(
    ntuple_path: pathlib.Path,
    pos_in_file: str,
    variable: str,
    bins: np.ndarray,
    weight: Optional[str] = None,
    selection_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """read an ntuple with uproot, the resulting arrays are then filled into a histogram

    Args:
        ntuple_path (pathlib.Path): path to ntuple
        pos_in_file (str): name of tree within ntuple
        variable (str): variable to bin histogram in
        bins (numpy.ndarray): bin edges for histogram
        weight (Optional[str], optional): event weight to extract, defaults to None (no weights applied)
        selection_filter (Optional[str], optional): filter to be applied on events, defaults to None (no filter)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - yield per bin
            - stat. uncertainty per bin
    """

    tree = uproot.open(str(ntuple_path) + ":" + pos_in_file)

    # extract observable and weights
    # need the last [variable] to select the right entry out of the dict
    # this only reads the observable branch and branches needed for the cut into memory
    observables = tree.arrays(variable, cut=selection_filter, library="np")[variable]

    if weight is not None:
        try:
            # if the weight is just a number, no branch needs to be read
            weights = np.ones_like(observables) * float(weight)
        except ValueError:
            # evaluate the expression with uproot4
            weights = tree.arrays(weight, cut=selection_filter, library="np")[weight]

    else:
        weights = np.ones_like(observables)

    yields, stdev = _bin_data(observables, weights, bins)
    return yields, stdev


def _bin_data(
    observables: np.ndarray, weights: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """create a histogram from unbinned data, providing yields, statistical uncertainties
    and the bin edges

    Args:
        observables (numpy.ndarray): values the histogram will be binned in
        weights (numpy.ndarray): weights to apply for each histogram entry
        bins (numpy.ndarray): bin edges for histogram

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - yield per bin
            - stat. uncertainty per bin
    """
    hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    hist.fill(observables, weight=weights)
    yields = hist.view().value
    stdev = np.sqrt(hist.view().variance)
    return yields, stdev
