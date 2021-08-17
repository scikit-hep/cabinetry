import logging

import numpy as np
import pytest

from cabinetry import smooth


@pytest.mark.parametrize(
    "orig, mod",
    [
        (([1, 3, 2, 5], 4), [1, 2, 3, 5]),
        (([1, 3, 2, 5, 11, 3, 8], 7), [1, 2, 3, 5, 5, 8, 8]),
    ],
)
def test__medians_353(orig, mod):
    smooth._medians_353(orig[0], orig[1])
    assert np.allclose(orig[0], mod)


def test_smooth_353qh_twice(caplog):
    caplog.set_level(logging.DEBUG)
    hist_smooth = smooth.smooth_353qh_twice([1, 3, 2, 5])
    assert isinstance(hist_smooth, list)
    assert np.allclose(hist_smooth, [1, 2, 3.25, 5])
    assert np.allclose(
        smooth.smooth_353qh_twice([1, 3, 2, 5, 11, 3, 8]),
        [1.0, 2.125, 3.66666675, 5.08333349, 6.16666651, 7.375, 8.0],
    )

    # with np.ndarray
    hist_smooth = smooth.smooth_353qh_twice(np.asarray([1, 3, 2, 5]))
    assert isinstance(hist_smooth, np.ndarray)
    assert hist_smooth.dtype == float
    assert np.allclose(hist_smooth, [1, 2, 3.25, 5])
    caplog.clear()

    assert np.allclose(smooth.smooth_353qh_twice([1, 3]), [1, 3])
    assert "at least three points needed for smoothing, no smoothing applied" in [
        rec.message for rec in caplog.records
    ]

    assert np.allclose(
        smooth.smooth_353qh_twice(
            [0.16505027, 0.92460960, 0.04765633, 0.36997846, 0.37872216]
        ),
        [0.16505027, 0.21628231, 0.31874642, 0.36997846, 0.36997846],
    )

    assert np.allclose(
        smooth.smooth_353qh_twice(
            [0.29093575, 0.93474650, 0.29723760, 0.61081952, -1.00000000]
        ),
        [0.29093575, 0.52901053, 0.47264278, -0.06587567, -1.00000000],
    )
    caplog.clear()
