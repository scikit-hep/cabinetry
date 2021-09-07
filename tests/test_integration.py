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
    model, data = cabinetry.model_utils.model_and_data(ws)
    fit_results = cabinetry.fit.fit(
        model, data, minos="Signal_norm", goodness_of_fit=True
    )

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
        0.04108188,
        0.03792561,
        0.03651562,
        0.04250561,
        0.93885915,
        0.99129285,
        0.55547829,
        0.62723287,
    ]
    best_twice_nll_expected = 17.194205
    corr_mat_expected = [
        [
            1.0,
            0.06280504,
            0.01952484,
            -0.02358537,
            -0.11133319,
            0.01074203,
            0.16261406,
            0.06767335,
        ],
        [
            0.06280504,
            1.0,
            0.00906325,
            -0.0049263,
            0.14681713,
            -0.02137699,
            -0.12260987,
            -0.18879984,
        ],
        [
            0.01952484,
            0.00906325,
            1.0,
            0.06545427,
            -0.1151505,
            0.00170718,
            0.0463534,
            0.03275504,
        ],
        [
            -0.02358537,
            -0.00492630,
            0.06545427,
            1.0,
            -0.05861778,
            -0.00075580,
            -0.04290516,
            -0.02539973,
        ],
        [
            -0.11133319,
            0.14681713,
            -0.1151505,
            -0.05861778,
            1.0,
            -0.16827929,
            -0.93117822,
            -0.8914587,
        ],
        [
            0.01074203,
            -0.02137699,
            0.00170718,
            -0.00075580,
            -0.16827929,
            1.0,
            0.05351298,
            -0.13751410,
        ],
        [
            0.16261406,
            -0.12260987,
            0.0463534,
            -0.04290516,
            -0.93117822,
            0.05351298,
            1.0,
            0.94146283,
        ],
        [
            0.06767335,
            -0.18879984,
            0.03275504,
            -0.02539973,
            -0.8914587,
            -0.1375141,
            0.94146283,
            1.0,
        ],
    ]
    assert np.allclose(fit_results.bestfit, bestfit_expected)
    assert np.allclose(fit_results.uncertainty, uncertainty_expected, atol=5e-5)
    assert np.allclose(fit_results.best_twice_nll, best_twice_nll_expected)
    assert np.allclose(fit_results.corr_mat, corr_mat_expected, rtol=1e-4, atol=5e-5)
    assert np.allclose(fit_results.goodness_of_fit, 0.24679341)

    # minos result
    assert "Signal_norm                =  1.6895 -0.9580 +0.9052" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # pre- and post-fit yield uncertainties
    model_prefit = cabinetry.model_utils.prediction(model)
    assert np.allclose(
        model_prefit.total_stdev_model_bins,
        [[69.040789, 58.343328, 38.219599, 45.296964]],
    )
    assert np.allclose(model_prefit.total_stdev_model_channels, [136.791978])
    _ = cabinetry.visualize.data_mc(model_prefit, data, close_figure=True)

    model_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
    assert np.allclose(
        model_postfit.total_stdev_model_bins,
        [[11.898333, 7.283185, 7.414715, 7.687922]],
        rtol=1e-4,
    )
    assert np.allclose(model_postfit.total_stdev_model_channels, [20.329523])
    _ = cabinetry.visualize.data_mc(model_postfit, data, close_figure=True)

    # nuisance parameter ranking
    ranking_results = cabinetry.fit.ranking(
        model, data, fit_results=fit_results, custom_fit=True
    )
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
            -0.10503361,
            0.13866138,
            -0.10801323,
            -0.05485056,
            -0.14842105,
            -1.10138138,
            -0.89692663,
        ],
        atol=5e-5,
    )
    assert np.allclose(
        ranking_results.postfit_down,
        [
            0.10604551,
            -0.14654595,
            0.11081914,
            0.05779502,
            0.1693045,
            0.84367903,
            0.76574073,
        ],
        atol=5e-5,
    )

    # parameter scan
    scan_results = cabinetry.fit.scan(
        model,
        data,
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
    limit_results = cabinetry.fit.limit(model, data, bracket=(0.5, 3.5), tolerance=0.05)
    assert np.allclose(limit_results.observed_limit, 3.1502, rtol=1e-2)
    assert np.allclose(
        limit_results.expected_limit,
        [1.0054, 1.3975, 1.9689, 2.7174, 3.5426],
        rtol=1e-2,
    )

    # discovery significance
    significance_results = cabinetry.fit.significance(model, data)
    np.allclose(significance_results.observed_p_value, 0.03583662)
    np.allclose(significance_results.observed_significance, 1.80118813)
    np.allclose(significance_results.expected_p_value, 0.14775040)
    np.allclose(significance_results.expected_significance, 1.04613046)
