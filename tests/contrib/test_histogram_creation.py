import numpy as np

from cabinetry.contrib import histogram_creation


def test__sumw2():
    weights = np.array([0.1, 0.2, 0.1])
    assert np.allclose(histogram_creation._sumw2(weights), 0.2449489743)


def test__bin_data():
    data = [1.1, 2.2, 2.9, 2.5, 1.4]
    weights = [1.0, 1.1, 0.9, 0.8, 1.5]
    bins = [1, 2, 3]
    yields, sumw2 = histogram_creation._bin_data(data, weights, bins)
    assert np.allclose(yields, [2.5, 2.8])
    assert np.allclose(sumw2, [1.80277564, 1.63095064])
