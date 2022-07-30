"""Creates histograms from ntuples with uproot."""

import pathlib
from typing import List, Optional

import boost_histogram as bh
import numpy as np
import uproot


def with_uproot(
    ntuple_paths: List[pathlib.Path],
    pos_in_file: str,
    variable: str,
    bins: np.ndarray,
    *,
    weight: Optional[str] = None,
    selection_filter: Optional[str] = None,
) -> bh.Histogram:
    """Reads an ntuple with uproot, fills and returns a histogram with the observable.

    The paths may contain wildcards.

    Args:
        ntuple_paths (List[pathlib.Path]): list of paths to ntuples
        pos_in_file (str): name of tree within ntuple
        variable (str): variable to bin histogram in
        bins (np.ndarray): bin edges for histogram
        weight (Optional[str], optional): event weight to extract, defaults to None (no
            weights applied)
        selection_filter (Optional[str], optional): filter to be applied on events,
            defaults to None (no filter)

    Returns:
        bh.Histogram: histogram containing data
    """
    # concatenate the path to file and location within file with ":"
    paths_with_trees = [str(path) + ":" + pos_in_file for path in ntuple_paths]

    # determine whether the weight is a float or an expression
    # (for which a branch needs to be read)
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
            paths_with_trees, expressions=[variable, weight], cut=selection_filter
        )
        obs_list = []
        weight_list = []
        for arr in array_generator:
            obs_list.append(arr[variable].to_numpy())
            weight_list.append(arr[weight].to_numpy())
        observables = np.concatenate(obs_list)
        weights = np.concatenate(weight_list)

    else:
        # only need to read the observables
        array_generator = uproot.iterate(
            paths_with_trees, expressions=[variable], cut=selection_filter
        )
        obs_list = []
        for arr in array_generator:
            obs_list.append(arr[variable].to_numpy())
        observables = np.concatenate(obs_list)
        weights = np.ones_like(observables) * float(weight)

    # create and return histogram
    hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    hist.fill(observables, weight=weights)
    return hist
