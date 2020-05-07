import numpy as np
import uproot


def from_uproot(ntuple_path, pos_in_file, selection, weight, bins, range):
    """
    create a single histogram with uproot
    """
    with uproot.open(ntuple_path) as f:
        events = f[pos_in_file].array(selection)
        if weight is not None:
            weights = f[pos_in_file].array(weight)
        else:
            weights = np.ones_like(events)

    histogram = _bin_data(events, weights, bins, range)
    return histogram


def _bin_data(data, weights, bins, bin_range=None):
    """
    create a histogram from unbinned data
    """
    if bin_range is not None:
        range_low, range_high = bin_range
        bins = np.linspace(range_low, range_high, bins)
    binned_data, _ = np.histogram(data, bins, bin_range, weights=weights)
    return binned_data
