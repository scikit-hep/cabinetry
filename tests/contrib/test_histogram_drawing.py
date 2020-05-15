import numpy as np

from cabinetry.contrib import histogram_drawing


def test__total_yield_uncertainty():
    sumw2_list = [[0.1, 0.2, 0.1], [0.3, 0.2, 0.1]]
    expected_uncertainties = [0.31622777, 0.28284271, 0.14142136]
    assert np.allclose(
        histogram_drawing._total_yield_uncertainty(sumw2_list), expected_uncertainties
    )
