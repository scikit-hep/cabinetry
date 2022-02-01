"""Creates histograms used as input for a minimal example of cabinetry."""
import os

import boost_histogram as bh
import numpy as np
import uproot


def run(output_directory):
    bins = [200.0, 300.0, 400.0, 500.0, 600.0]
    hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    file_name = output_directory + "/histograms.root"
    with uproot.recreate(file_name) as f:
        yields = np.asarray([112.0, 112.0, 124.0, 66.0])
        stdev = np.asarray([10.58300524, 10.58300524, 11.13552873, 8.1240384])
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        f["SR/Data/Nominal"] = hist

        yields = np.asarray([0.0, 1.58536913, 23.6164268, 24.54892223])
        stdev = np.asarray([0.0, 0.19931166, 0.77410713, 0.78830503])
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        f["SR/Signal/Nominal"] = hist

        yields = np.asarray([112.73896936, 128.62169539, 88.10700838, 55.24607072])
        stdev = np.asarray([4.76136678, 5.10645036, 4.21104367, 3.34933335])
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        f["SR/Background/Nominal"] = hist

        yields = np.asarray([53.85246451, 85.0382603, 90.75880537, 78.14459379])
        stdev = np.asarray([3.29419354, 4.14450981, 4.27730978, 3.96701472])
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        f["SR/Background/Modeling_Up"] = hist

        yields = np.asarray([98.13569365, 154.1222757, 135.20449815, 103.14744392])
        stdev = np.asarray([4.17025569, 6.14088444, 6.47920695, 6.2581315])
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        f["SR/Background/WeightBasedModeling_Up"] = hist

        yields = np.asarray([78.91727855, 90.03518677, 61.67490587, 38.6722495])
        stdev = np.asarray([3.33295675, 3.57451525, 2.94773057, 2.34453334])
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        f["SR/Background/WeightBasedModeling_Down"] = hist


if __name__ == "__main__":
    output_directory = "inputs/"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    run(output_directory)
