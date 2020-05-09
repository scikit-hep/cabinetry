import numpy as np
from scipy import stats
import uproot


def from_uproot(ntuple_path, pos_in_file, selection, weight, bins):
    """
    create a single histogram with uproot, return bin yields, statistical
    uncertainties on the yield, and bin edges
    """
    with uproot.open(ntuple_path) as f:
        events = f[pos_in_file].array(selection)
        if weight is not None:
            weights = f[pos_in_file].array(weight)
        else:
            weights = np.ones_like(events)

    yields, sumw2 = _bin_data(events, weights, bins)
    return yields, sumw2


def _sumw2(arr):
    """
    calculate the absolute statistical uncertainty given an array of weights in a bin
    """
    return np.sqrt(np.sum(arr * arr))


def _bin_data(data, weights, bins):
    """
    create a histogram from unbinned data, providing yields, statistical uncertainties
    and the bin edges
    """
    # get bin yield and stat. uncertainty
    yields, _, _ = stats.binned_statistic(data, weights, statistic="sum", bins=bins)
    sumw2, _, _ = stats.binned_statistic(data, weights, statistic=_sumw2, bins=bins)
    return yields, sumw2
