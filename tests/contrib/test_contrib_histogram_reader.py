import numpy as np

from cabinetry.contrib import histogram_reader


def test_with_uproot(tmp_path, utils):
    fname = tmp_path / "test.root"
    histname = "h1"
    bins = np.asarray([1, 2, 3])
    yields = np.asarray([1.0, 2.0])
    stdev = np.asarray([0.1, 0.2])
    # create something to read
    utils.create_histogram(fname, histname, bins, yields, stdev)

    hist = histogram_reader.with_uproot(f"{fname}:{histname}")
    assert np.allclose(hist.axes[0].edges, bins)
    assert np.allclose(hist.values(), yields)
    assert np.allclose(np.sqrt(hist.variances()), stdev)
