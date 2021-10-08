"""Reads histograms with uproot."""

import boost_histogram as bh
import uproot


def with_uproot(histo_path: str) -> bh.Histogram:
    """Reads a histogram with uproot and returns it.

    Args:
        histo_path (str): path to histogram, use a colon to distinguish between path to
            file and path to histogram within file (example: ``file.root:h1``)

    Returns:
        bh.Histogram: histogram containing data
    """
    hist = uproot.open(histo_path).to_boost()
    return hist
