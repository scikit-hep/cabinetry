import numpy as np
import pytest

from cabinetry import histo
from cabinetry import template_postprocessor


@pytest.mark.parametrize(
    "test_histo, fixed_sumw2",
    [
        (histo.Histogram([], [float("nan"), 0.2], []), [0.0, 0.2]),
        (histo.Histogram([], [0.1, 0.2], []), [0.1, 0.2]),
    ],
)
def test__fix_stat_unc(test_histo, fixed_sumw2):
    name = "test_histo"
    template_postprocessor._fix_stat_unc(test_histo, name)
    assert np.allclose(test_histo.sumw2, fixed_sumw2)


def test_apply_postprocessing():
    histogram = histo.Histogram([], [float("nan"), 0.2], [])
    name = "test_histo"
    fixed_sumw2 = [0.0, 0.2]
    fixed_histogram = template_postprocessor.apply_postprocessing(histogram, name)
    assert np.allclose(fixed_histogram.sumw2, fixed_sumw2)
    # the original histogram should be unchanged
    np.testing.assert_equal(histogram.sumw2, [float("nan"), 0.2])


def test_run(tmp_path):
    config = {"Samples": [{"Name": "signal"}], "Regions": [{"Name": "region_1"}]}

    # create an input histogram
    histo_path = tmp_path / "signal_region_1_nominal.npz"
    histogram = histo.Histogram(
        np.asarray([1.0, 2.0]), np.asarray([1.0, 1.0]), np.asarray([0.0, 1.0, 2.0])
    )
    histogram.save(histo_path)

    template_postprocessor.run(config, tmp_path)
    modified_histo = histo.Histogram.from_path(histo_path, modified=True)
    assert np.allclose(modified_histo.yields, histogram.yields)
    assert np.allclose(modified_histo.sumw2, histogram.sumw2)
    assert np.allclose(modified_histo.bins, histogram.bins)
