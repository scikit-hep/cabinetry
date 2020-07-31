import numpy as np

from cabinetry.contrib import histogram_creation


def test_from_uproot(tmp_path, utils):
    fname = tmp_path / "test.root"
    treename = "tree"
    varname = "var"
    var_array = [1.1, 2.3, 3.0, 3.2]
    weightname_write = "weight"
    weight_array = [1.0, 1.0, 2.0, 1.0]
    bins = np.asarray([1, 2, 3, 4])
    # create something to read
    utils.create_ntuple(
        fname, treename, varname, var_array, weightname_write, weight_array
    )

    # read - no weights or selection
    yields, stdev = histogram_creation.from_uproot(fname, treename, varname, bins)
    assert np.allclose(yields, [1, 1, 2])
    assert np.allclose(stdev, [1, 1, 1.41421356])

    # read - with selection cut
    selection = "var < 3.1"
    yields, stdev = histogram_creation.from_uproot(
        fname, treename, varname, bins, selection_filter=selection
    )
    assert np.allclose(yields, [1, 1, 1])
    assert np.allclose(stdev, [1, 1, 1])

    # read - with weight
    weightname_apply = "weight*2"
    yields, stdev = histogram_creation.from_uproot(
        fname, treename, varname, bins, weight=weightname_apply
    )
    assert np.allclose(yields, [2, 2, 6])
    assert np.allclose(stdev, [2, 2, 4.47213595])

    # read - with weight that is only a float
    weight_float = "2.0"
    yields, stdev = histogram_creation.from_uproot(
        fname, treename, varname, bins, weight=weight_float
    )
    assert np.allclose(yields, [2, 2, 4])
    assert np.allclose(stdev, [2, 2, 2.82842712])


def test__bin_data():
    data = np.asarray([1.1, 2.2, 2.9, 2.5, 1.4])
    weights = np.asarray([1.0, 1.1, 0.9, 0.8, 1.5])
    bins = np.asarray([1, 2, 3])
    yields, stdev = histogram_creation._bin_data(data, weights, bins)
    assert np.allclose(yields, [2.5, 2.8])
    assert np.allclose(stdev, [1.80277564, 1.63095064])
