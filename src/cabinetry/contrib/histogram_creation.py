import boost_histogram as bh
import numexpr as ne
import numpy as np
import uproot


def from_uproot(
    ntuple_path, pos_in_file, variable, bins, weight=None, selection_filter=None
):
    """create a single histogram with uproot, return bin yields and statistical
    uncertainties on the yield

    Args:
        ntuple_path (pathlib.Path): path to ntuple
        pos_in_file (str): name of tree within ntuple
        variable (str): variable to bin histogram in
        weight (str): event weight to extract
        bins (numpy.ndarray): bin edges for histogram
        selection_filter (str, optional): filter to be applied on events, defaults to None (no filter)

    Returns:
        (numpy.ndarray, numpy.ndarray): tuple of yields and stat. uncertainties for all bins
    """
    table = uproot.open(ntuple_path)[pos_in_file].lazyarrays()

    # extract observable and weights
    observables = table[variable]
    if weight is not None:
        weights = table[weight]
    else:
        weights = np.ones_like(observables)

    # filter events if requested
    if selection_filter is not None:
        selection_mask = ne.evaluate(selection_filter, table)
        observables = observables[selection_mask]
        weights = weights[selection_mask]

    # convert everything into numpy ndarrays - probably not needed in general
    # and there might be a better solution
    observables = np.asarray(observables)
    weights = np.asarray(weights)

    yields, sumw2 = _bin_data(observables, weights, bins)
    return yields, sumw2


def _bin_data(observables, weights, bins):
    """create a histogram from unbinned data, providing yields, statistical uncertainties
    and the bin edges

    Args:
        observables (numpy.ndarray): values the histogram will be binned in
        weights (numpy.ndarray): weights to apply for each histogram entry
        bins (numpy.ndarray): bin edges for histogram

    Returns:
        tuple: a tuple containing
            - numpy.ndarray: yields per bin
            - numpy.ndarray: and stat. uncertainty per bin
    """
    hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    hist.fill(observables, weight=weights)
    yields = hist.view().value
    sumw2 = np.sqrt(hist.view().variance)
    return yields, sumw2
