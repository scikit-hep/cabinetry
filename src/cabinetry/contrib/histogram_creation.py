import pathlib
from typing import Optional, Tuple

import awkward1 as ak
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
    # if there is more than one file handed over in the list, need to concatenate differently with ":"

    # determine whether the weight is a float or an expression (for which a branch needs to be read)
    if weight is not None:
        try:
            float(weight)
            weight_is_expression = False
        except ValueError:
            # weight is not a float, need to evaluate the expression
            weight_is_expression = True
    else:
        # no weight specified, all weights are 1.0
        weight_is_expression = False
        weight = "1.0"

    if weight_is_expression:
        # need to read observables and weights
        array_generator = uproot.iterate(
            str(ntuple_path) + ":" + pos_in_file,
            expressions=[variable, weight],
            cut=selection_filter,
        )
        obs_list = []
        weight_list = []
        for arr in array_generator:
            obs_list.append(ak.to_numpy(arr[variable]))
            weight_list.append(ak.to_numpy(arr[weight]))
        observables = np.concatenate(obs_list)
        weights = np.concatenate(weight_list)

    else:
        # only need to read the observables
        array_generator = uproot.iterate(
            str(ntuple_path) + ":" + pos_in_file,
            expressions=[variable],
            cut=selection_filter,
        )
        obs_list = []
        for arr in array_generator:
            obs_list.append(ak.to_numpy(arr[variable]))
        observables = np.concatenate(obs_list)
        weights = np.ones_like(observables) * float(weight)

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
