from matplotlib.testing.compare import compare_images
import numpy as np

from cabinetry.contrib import histogram_drawing
from cabinetry import histo


def test__total_yield_uncertainty():
    sumw2_list = [[0.1, 0.2, 0.1], [0.3, 0.2, 0.1]]
    expected_uncertainties = [0.31622777, 0.28284271, 0.14142136]
    assert np.allclose(
        histogram_drawing._total_yield_uncertainty(sumw2_list), expected_uncertainties
    )


def test_data_MC_matplotlib(tmp_path):
    fname = tmp_path / "subdir" / "fig.pdf"
    bg_hist = {
        "yields": np.asarray([12.5, 14]),
        "sumw2": np.asarray([0.4, 0.5]),
        "bins": np.asarray([1, 2, 3]),
    }
    sig_hist = {
        "yields": np.asarray([2, 5]),
        "sumw2": np.asarray([0.1, 0.2]),
        "bins": np.asarray([1, 2, 3]),
    }
    data_hist = {
        "yields": np.asarray([13, 15]),
        "sumw2": np.asarray([3.61, 3.87]),
        "bins": np.asarray([1, 2, 3]),
    }
    histo_dict_list = [
        {"label": "Background", "isData": False, "hist": bg_hist, "variable": "x"},
        {"label": "Signal", "isData": False, "hist": sig_hist, "variable": "x"},
        {"label": "Data", "isData": True, "hist": data_hist, "variable": "x"},
    ]
    histogram_drawing.data_MC_matplotlib(histo_dict_list, fname)
    assert (
        compare_images("tests/contrib/reference/ref_data_MC.pdf", str(fname), 0) is None
    )
