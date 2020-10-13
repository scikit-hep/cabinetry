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
    fit_results = cabinetry.fit.fit(ws, minos_parameters="Signal_norm")

    bestfit_expected = [
        1.00097604,
        0.98918703,
        1.01954935,
        0.98300027,
        1.68652611,
        -0.09142767,
        -0.32313458,
        -0.58374164,
    ]
    uncertainty_expected = [
        0.0410808,
        0.03792673,
        0.03651597,
        0.04250513,
        0.92411694,
        0.99112844,
        0.55537361,
        0.627469,
    ]
    best_twice_nll_expected = 17.194286
    corr_mat_expected = [
        [
            1.0,
            0.0628096592,
            0.0195177192,
            -0.0235685409,
            -0.111109465,
            0.0106844629,
            0.162391048,
            0.0674859362,
        ],
        [
            0.0628096592,
            1.0,
            0.00904907811,
            -0.0049250707,
            0.147074209,
            -0.0213145107,
            -0.122847225,
            -0.189009516,
        ],
        [
            0.0195177192,
            0.00904907811,
            1.0,
            0.0654356196,
            -0.11511303,
            0.00166640856,
            0.0463716356,
            0.0327822936,
        ],
        [
            -0.0235685409,
            -0.0049250707,
            0.0654356196,
            1.0,
            -0.0586033381,
            -0.000760862587,
            -0.0428518642,
            -0.0253515109,
        ],
        [
            -0.111109465,
            0.147074209,
            -0.11511303,
            -0.0586033381,
            1.0,
            -0.167610909,
            -0.93124935,
            -0.891613116,
        ],
        [
            0.0106844629,
            -0.0213145107,
            0.00166640856,
            -0.000760862587,
            -0.167610909,
            1.0,
            0.0530075596,
            -0.137965149,
        ],
        [
            0.162391048,
            -0.122847225,
            0.0463716356,
            -0.0428518642,
            -0.93124935,
            0.0530075596,
            1.0,
            0.941499622,
        ],
        [
            0.0674859362,
            -0.189009516,
            0.0327822936,
            -0.0253515109,
            -0.891613116,
            -0.137965149,
            0.941499622,
            1.0,
        ],
    ]
    assert np.allclose(fit_results.bestfit, bestfit_expected)
    assert np.allclose(fit_results.uncertainty, uncertainty_expected)
    assert np.allclose(fit_results.best_twice_nll, best_twice_nll_expected)
    assert np.allclose(fit_results.corr_mat, corr_mat_expected, rtol=5e-5)

    # minos result
    assert "Signal_norm                    = 1.6865 -0.9551 +0.9083" in [
        rec.message for rec in caplog.records
    ]

    # nuisance parameter ranking
    ranking_results = cabinetry.fit.ranking(ws, fit_results)
    assert np.allclose(
        ranking_results.prefit_up,
        [
            -0.10470340,
            0.14648429,
            -0.10991236,
            -0.05278778,
            -0.14582642,
            -1.67590283,
            -1.51782965,
        ],
    )
    assert np.allclose(
        ranking_results.prefit_down,
        [
            0.11277466,
            -0.14829532,
            0.11949235,
            0.06167027,
            0.17465462,
            1.41673474,
            1.15103106,
        ],
    )
    assert np.allclose(
        ranking_results.postfit_up,
        [
            -0.10184155,
            0.14165853,
            -0.10464174,
            -0.05181179,
            -0.14457961,
            -1.10093621,
            -0.89766455,
        ],
    )
    assert np.allclose(
        ranking_results.postfit_down,
        [
            0.1097976,
            -0.14270861,
            0.11393967,
            0.06079421,
            0.17307896,
            0.84412165,
            0.76660498,
        ],
    )

    # parameter scan
    scan_results = cabinetry.fit.scan(
        ws, "Signal_norm", par_range=(1.18967971, 2.18967971), n_steps=3
    )
    # lower edge of scan is beyond normalization factor bounds specified in workspace
    assert np.allclose(
        scan_results.delta_nlls, [0.27154057, 0.0, 0.29018711], atol=5e-5
    )
