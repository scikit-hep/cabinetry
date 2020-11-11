import logging

import numpy as np
import pytest

import cabinetry
from util import create_ntuples


@pytest.fixture
def ntuple_creator():
    return create_ntuples.run


@pytest.mark.no_cover
def test_integration(tmp_path, ntuple_creator, caplog):
    """The purpose of this integration test is to check whether the
    steps run without error and whether the fit result is as expected.
    """
    ntuple_creator(str(tmp_path))

    cabinetry_config = cabinetry.configuration.load("config_example.yml")

    # override config options to point to tmp_path
    cabinetry_config["General"]["HistogramFolder"] = str(tmp_path / "histograms")
    cabinetry_config["General"]["InputPath"] = str(tmp_path / "{SamplePaths}")

    caplog.set_level(logging.DEBUG)

    cabinetry.template_builder.create_histograms(cabinetry_config, method="uproot")
    cabinetry.template_postprocessor.run(cabinetry_config)
    workspace_path = tmp_path / "example_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config)
    cabinetry.workspace.save(ws, workspace_path)
    ws = cabinetry.workspace.load(workspace_path)
    fit_results = cabinetry.fit.fit(ws, minos="Signal_norm", goodness_of_fit=True)

    bestfit_expected = [
        1.00102289,
        0.98910429,
        1.01970061,
        0.98296235,
        1.68953317,
        -0.0880332,
        -0.32457145,
        -0.58582788,
    ]
    uncertainty_expected = [
        0.04108053,
        0.03792499,
        0.03651557,
        0.04250593,
        0.93760399,
        0.99131386,
        0.5547287,
        0.62647293,
    ]
    best_twice_nll_expected = 17.194205
    corr_mat_expected = [
        [
            1.0,
            0.06287511,
            0.01949609,
            -0.02357913,
            -0.11102759,
            0.01070953,
            0.16237686,
            0.06730786,
        ],
        [
            0.06287511,
            1.0,
            0.00909003,
            -0.00493023,
            0.14668324,
            -0.02131046,
            -0.12244755,
            -0.18872808,
        ],
        [
            0.01949609,
            0.00909003,
            1.0,
            0.06546918,
            -0.11512695,
            0.00171275,
            0.04622224,
            0.03260444,
        ],
        [
            -0.02357913,
            -0.00493023,
            0.06546918,
            1.0,
            -0.05875069,
            -0.00072121,
            -0.04292646,
            -0.02540061,
        ],
        [
            -0.11102759,
            0.14668324,
            -0.11512695,
            -0.05875069,
            1.0,
            -0.1681882,
            -0.93098096,
            -0.89116314,
        ],
        [
            0.01070953,
            -0.02131046,
            0.00171275,
            -0.00072121,
            -0.1681882,
            1.0,
            0.0532316,
            -0.13803459,
        ],
        [
            0.16237686,
            -0.12244755,
            0.04622224,
            -0.04292646,
            -0.93098096,
            0.0532316,
            1.0,
            0.94131251,
        ],
        [
            0.06730786,
            -0.18872808,
            0.03260444,
            -0.02540061,
            -0.89116314,
            -0.13803459,
            0.94131251,
            1.0,
        ],
    ]
    assert np.allclose(fit_results.bestfit, bestfit_expected)
    assert np.allclose(fit_results.uncertainty, uncertainty_expected)
    assert np.allclose(fit_results.best_twice_nll, best_twice_nll_expected)
    assert np.allclose(fit_results.corr_mat, corr_mat_expected, rtol=1e-4)
    assert np.allclose(fit_results.goodness_of_fit, 0.24679341)

    # minos result
    assert "Signal_norm                    = 1.6895 -0.9580 +0.9052" in [
        rec.message for rec in caplog.records
    ]

    # nuisance parameter ranking
    ranking_results = cabinetry.fit.ranking(ws, fit_results, custom_fit=True)
    assert np.allclose(
        ranking_results.prefit_up,
        [
            -0.10787925,
            0.14350015,
            -0.11328328,
            -0.05573699,
            -0.14964691,
            -1.67766172,
            -1.51712811,
        ],
    )
    assert np.allclose(
        ranking_results.prefit_down,
        [
            0.10910502,
            -0.15092356,
            0.1162032,
            0.05864963,
            0.17084682,
            1.41570164,
            1.15025176,
        ],
    )
    assert np.allclose(
        ranking_results.postfit_up,
        [
            -0.10503025,
            0.13865911,
            -0.10801309,
            -0.05485107,
            -0.14842401,
            -1.09991789,
            -0.89574803,
        ],
    )
    assert np.allclose(
        ranking_results.postfit_down,
        [
            0.10604367,
            -0.14654339,
            0.11081899,
            0.05779548,
            0.16930794,
            0.84262594,
            0.76490797,
        ],
    )

    # parameter scan
    scan_results = cabinetry.fit.scan(
        ws,
        "Signal_norm",
        par_range=(1.18967971, 2.18967971),
        n_steps=3,
        custom_fit=True,
    )
    # lower edge of scan is beyond normalization factor bounds specified in workspace
    assert np.allclose(
        scan_results.delta_nlls, [0.27153966, 0.0, 0.29018620], atol=5e-5
    )

    # upper limit, this calculation is slow
    limit_results = cabinetry.fit.limit(ws, bracket=[0.5, 3.5], tolerance=0.05)
    assert np.allclose(limit_results.observed_limit, 3.2295, rtol=1e-2)
    assert np.allclose(
        limit_results.expected_limit,
        [1.0464, 1.4309, 1.8968, 2.6627, 3.4603],
        rtol=1e-2,
    )
