import numpy as np

from cabinetry.contrib import histogram_creator


def test_with_uproot(tmp_path, utils):
    fnames = [tmp_path / "test.root"]
    treename = "tree"
    varname = "var"
    var_array = np.asarray([1.1, 2.3, 3.0, 3.2])
    weightname_write = "weight"
    weight_array = np.asarray([1.0, 1.0, 2.0, 1.0])
    bins = np.asarray([1, 2, 3, 4])
    # create something to read
    utils.create_ntuple(
        fnames[0], treename, varname, var_array, weightname_write, weight_array
    )

    # read - no weights or selection
    hist = histogram_creator.with_uproot(fnames, treename, varname, bins)
    assert np.allclose(hist.values(), [1, 1, 2])
    assert np.allclose(np.sqrt(hist.variances()), [1, 1, 1.41421356])

    # read - with selection cut
    selection = "var < 3.1"
    hist = histogram_creator.with_uproot(
        fnames, treename, varname, bins, selection_filter=selection
    )
    assert np.allclose(hist.values(), [1, 1, 1])
    assert np.allclose(np.sqrt(hist.variances()), [1, 1, 1])

    # read - with weight
    weightname_apply = "weight*2"
    hist = histogram_creator.with_uproot(
        fnames, treename, varname, bins, weight=weightname_apply
    )
    assert np.allclose(hist.values(), [2, 2, 6])
    assert np.allclose(np.sqrt(hist.variances()), [2, 2, 4.47213595])

    # read - with weight that is only a float
    weight_float = "2.0"
    hist = histogram_creator.with_uproot(
        fnames, treename, varname, bins, weight=weight_float
    )
    assert np.allclose(hist.values(), [2, 2, 4])
    assert np.allclose(np.sqrt(hist.variances()), [2, 2, 2.82842712])

    # create an additional file to test wildcards / input lists
    extra_path = tmp_path / "test_2.root"
    utils.create_ntuple(
        extra_path, treename, varname, var_array, weightname_write, weight_array
    )

    # read - with wildcard, matching two files
    fnames = [tmp_path / "test*.root"]
    hist = histogram_creator.with_uproot(fnames, treename, varname, bins)
    assert np.allclose(hist.values(), [2, 2, 4])
    assert np.allclose(np.sqrt(hist.variances()), [1.41421356, 1.41421356, 2])

    # read - with two list entries
    fnames = [tmp_path / "test.root", tmp_path / "test_2.root"]
    hist = histogram_creator.with_uproot(fnames, treename, varname, bins)
    assert np.allclose(hist.values(), [2, 2, 4])
    assert np.allclose(np.sqrt(hist.variances()), [1.41421356, 1.41421356, 2])
