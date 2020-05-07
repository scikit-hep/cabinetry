"""
it might make sense to allow serialization of histograms in various
different formats, so saving and loading should go through this wrapper
"""
import os

import numpy as np


def save_histogram(histogram, path, histogram_name):
    """
    save a histogram to disk
    """
    print("- saving", histogram_name, "to", path)

    # create output directory if it does not exist yet
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path + histogram_name + ".npy", histogram)


def load_histogram(path, histogram_name):
    """
    load a histogram from disk
    """
    histogram = np.load(path + histogram_name + ".npy")
    return histogram


def _build_histogram_name(sample, region, systematic):
    """
    construct a unique name for each histogram
    param sample: the sample
    """
    name = sample["Name"] + "_" + region["Name"] + "_" + systematic["Name"]
    name = name.replace(" ", "-")
    return name
