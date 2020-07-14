import boost_histogram as bh
import numpy as np
import uproot4 as uproot


def from_uproot(
    ntuple_path, pos_in_file, variable, bins, weight=None, selection_filter=None
):
    """read an ntuple with uproot, the resulting arrays are then filled into a histogram

    Args:
        ntuple_path (pathlib.Path): path to ntuple
        pos_in_file (str): name of tree within ntuple
        variable (str): variable to bin histogram in
        bins (numpy.ndarray): bin edges for histogram
        weight (str): event weight to extract, defaults to None (no weights applied)
        selection_filter (str, optional): filter to be applied on events, defaults to None (no filter)

    Returns:
        (numpy.ndarray, numpy.ndarray): tuple of yields and stat. uncertainties for all bins
    """
    tree = uproot.open(str(ntuple_path) + ":" + pos_in_file)

    # extract observable and weights
    # need the last [variable] to select the right entry out of the dict
    # this only reads the observable branch and branches needed for the cut into memory
    observables = tree[variable].arrays(cut=selection_filter, library="np")[variable]

    if weight is not None:
        weights = tree[weight].arrays(cut=selection_filter, library="np")[weight]
    else:
        weights = np.ones_like(observables)

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
