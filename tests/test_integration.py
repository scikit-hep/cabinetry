import logging

import numpy as np
import pytest

import cabinetry
from utils import create_histograms, create_ntuples


@pytest.fixture
def histogram_creator():
    return create_histograms.run


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
    cabinetry_config["General"]["InputPath"] = str(tmp_path / "{SamplePath}")

    caplog.set_level(logging.DEBUG)

    cabinetry.templates.build(cabinetry_config, method="uproot")
    cabinetry.templates.postprocess(cabinetry_config)
    workspace_path = tmp_path / "example_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config)
    cabinetry.workspace.save(ws, workspace_path)
    ws = cabinetry.workspace.load(workspace_path)
    model, data = cabinetry.model_utils.model_and_data(ws)
    fit_results = cabinetry.fit.fit(
        model, data, minos="Signal_norm", goodness_of_fit=True
    )

    bestfit_expected = [
        -0.32457145,
        -0.58582788,
        1.68953317,
        -0.0880332,
        1.00102289,
        0.98910429,
        1.01970061,
        0.98296235,
    ]
    uncertainty_expected = [
        0.55547829,
        0.62723287,
        0.93885915,
        0.99129285,
        0.04108188,
        0.03792561,
        0.03651562,
        0.04250561,
    ]
    best_twice_nll_expected = 17.194205
    corr_mat_expected = [
        [
            1.0,
            0.94146283,
            -0.93117822,
            0.05351298,
            0.16261406,
            -0.12260987,
            0.0463534,
            -0.04290516,
        ],
        [
            0.94146283,
            1.0,
            -0.8914587,
            -0.1375141,
            0.06767335,
            -0.18879984,
            0.03275504,
            -0.02539973,
        ],
        [
            -0.93117822,
            -0.8914587,
            1.0,
            -0.16827929,
            -0.11133319,
            0.14681713,
            -0.1151505,
            -0.05861778,
        ],
        [
            0.05351298,
            -0.13751410,
            -0.16827929,
            1.0,
            0.01074203,
            -0.02137699,
            0.00170718,
            -0.00075580,
        ],
        [
            0.16261406,
            0.06767335,
            -0.11133319,
            0.01074203,
            1.0,
            0.06280504,
            0.01952484,
            -0.02358537,
        ],
        [
            -0.12260987,
            -0.18879984,
            0.14681713,
            -0.02137699,
            0.06280504,
            1.0,
            0.00906325,
            -0.0049263,
        ],
        [
            0.0463534,
            0.03275504,
            -0.1151505,
            0.00170718,
            0.01952484,
            0.00906325,
            1.0,
            0.06545427,
        ],
        [
            -0.04290516,
            -0.02539973,
            -0.05861778,
            -0.00075580,
            -0.02358537,
            -0.00492630,
            0.06545427,
            1.0,
        ],
    ]
    assert np.allclose(fit_results.bestfit, bestfit_expected)
    assert np.allclose(fit_results.uncertainty, uncertainty_expected, atol=1e-4)
    assert np.allclose(fit_results.best_twice_nll, best_twice_nll_expected)
    assert np.allclose(fit_results.corr_mat, corr_mat_expected, rtol=1e-4, atol=5e-5)
    assert np.allclose(fit_results.goodness_of_fit, 0.24679341)

    # minos result
    assert "Signal_norm                =  1.6895 -0.9580 +0.9052" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # pre- and post-fit yield uncertainties
    prediction_prefit = cabinetry.model_utils.prediction(model)
    assert np.allclose(
        prediction_prefit.total_stdev_model_bins,
        [
            [
                [69.040789, 58.329118, 37.973787, 45.137157],  # background
                [0.0, 0.100772, 1.487773, 1.620867],  # signal
                [69.040789, 58.343328, 38.219599, 45.296964],  # sum over samples
            ]
        ],
    )
    assert np.allclose(
        prediction_prefit.total_stdev_model_channels,
        [[136.368732, 2.851565, 136.791978]],
    )
    _ = cabinetry.visualize.data_mc(prediction_prefit, data, close_figure=True)

    prediction_postfit = cabinetry.model_utils.prediction(
        model, fit_results=fit_results
    )
    assert np.allclose(
        prediction_postfit.total_stdev_model_bins,
        [
            [
                [11.898551, 7.513216, 21.002006, 24.284847],  # background
                [0.0, 1.467646, 22.137293, 22.269200],  # signal
                [11.898551, 7.283171, 7.414715, 7.687966],  # sum over samples
            ]
        ],
        rtol=1e-4,
    )
    assert np.allclose(
        prediction_postfit.total_stdev_model_channels,
        [[41.043814, 45.814417, 20.439575]],
        atol=5e-3,
    )
    _ = cabinetry.visualize.data_mc(prediction_postfit, data, close_figure=True)

    # nuisance parameter ranking
    ranking_results = cabinetry.fit.ranking(
        model, data, fit_results=fit_results, custom_fit=True
    )
    assert np.allclose(
        ranking_results.prefit_up,
        [
            -1.67617185,
            -1.51531885,
            -0.14964691,
            -0.11067885,
            0.14350015,
            -0.11311905,
            -0.05573699,
        ],
    )
    assert np.allclose(
        ranking_results.prefit_down,
        [
            1.41570164,
            1.15001063,
            0.16388633,
            0.10921275,
            -0.15331233,
            0.11492629,
            0.05864962,
        ],
    )
    assert np.allclose(
        ranking_results.postfit_up,
        [
            -1.10235942,
            -0.89697066,
            -0.14842092,
            -0.10774037,
            0.13866107,
            -0.10800505,
            -0.0548506,
        ],
        atol=1e-4,
    )
    assert np.allclose(
        ranking_results.postfit_down,
        [
            0.84366301,
            0.76572728,
            0.16236075,
            0.10604544,
            -0.14383307,
            0.10888311,
            0.05779505,
        ],
        atol=1e-4,
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
        scan_results.delta_nlls, [0.27163227, 0.0, 0.29018620], atol=5e-5
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


@pytest.mark.no_cover
def test_histogram_reading(tmp_path, histogram_creator):
    histogram_creator(str(tmp_path))

    cabinetry_config = cabinetry.configuration.load("utils/config_histograms.yml")

    # override config options to point to tmp_path
    cabinetry_config["General"]["HistogramFolder"] = str(tmp_path / "histograms")
    cabinetry_config["General"]["InputPath"] = (
        f"{tmp_path / 'histograms.root'}:" + "{RegionPath}/{SamplePath}/{VariationPath}"
    )

    cabinetry.templates.collect(cabinetry_config, method="uproot")
    cabinetry.templates.postprocess(cabinetry_config)
    ws = cabinetry.workspace.build(cabinetry_config)
    model, data = cabinetry.model_utils.model_and_data(ws)
    fit_results = cabinetry.fit.fit(
        model, data, minos="Signal_norm", goodness_of_fit=True
    )

    bestfit_expected = [
        -0.32457145,
        -0.58582788,
        1.68953317,
        -0.0880332,
        1.00102289,
        0.98910429,
        1.01970061,
        0.98296235,
    ]
    best_twice_nll_expected = 17.194205
    assert np.allclose(fit_results.bestfit, bestfit_expected)
    assert np.allclose(fit_results.best_twice_nll, best_twice_nll_expected)
