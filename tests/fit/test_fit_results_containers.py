import logging

import numpy as np

from cabinetry import fit


def test_FitResults():
    bestfit = np.asarray([1.0])
    uncertainty = np.asarray([0.1])
    labels = ["par_a"]
    corr_mat = np.asarray([[1.0]])
    best_twice_nll = 2.0
    fit_results = fit.FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)
    assert np.allclose(fit_results.bestfit, bestfit)
    assert np.allclose(fit_results.uncertainty, uncertainty)
    assert fit_results.labels == labels
    assert np.allclose(fit_results.corr_mat, corr_mat)
    assert fit_results.best_twice_nll == best_twice_nll
    assert fit_results.goodness_of_fit == -1
    assert fit_results.minos_uncertainty == {}


def test_RankingResults():
    bestfit = np.asarray([1.0])
    uncertainty = np.asarray([0.1])
    labels = ["par_a"]
    prefit_up = np.asarray([0.3])
    prefit_down = np.asarray([-0.3])
    postfit_up = np.asarray([0.2])
    postfit_down = np.asarray([-0.2])
    ranking_results = fit.RankingResults(
        bestfit, uncertainty, labels, prefit_up, prefit_down, postfit_up, postfit_down
    )
    assert np.allclose(ranking_results.bestfit, bestfit)
    assert np.allclose(ranking_results.uncertainty, uncertainty)
    assert ranking_results.labels == labels
    assert np.allclose(ranking_results.prefit_up, prefit_up)
    assert np.allclose(ranking_results.prefit_down, prefit_down)
    assert np.allclose(ranking_results.postfit_up, postfit_up)
    assert np.allclose(ranking_results.postfit_down, postfit_down)


def test_ScanResults():
    name = "par_a"
    bestfit = 1.2
    uncertainty = 0.3
    parameter_values = np.asarray([0.9, 1.2, 1.5])
    delta_nlls = np.asarray([1.0, 0.0, 1.0])
    scan_results = fit.ScanResults(
        name, bestfit, uncertainty, parameter_values, delta_nlls
    )
    assert scan_results.name == name
    assert scan_results.bestfit == bestfit
    assert scan_results.uncertainty == uncertainty
    assert np.allclose(scan_results.parameter_values, parameter_values)
    assert np.allclose(scan_results.delta_nlls, delta_nlls)


def test_LimitResults():
    observed_limit = 3.0
    expected_limit = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    observed_CLs = np.asarray([0.05])
    expected_CLs = np.asarray([0.01, 0.02, 0.05, 0.07, 0.10])
    poi_values = np.asarray([3.0])
    confidence_level = 0.90
    limit_results = fit.LimitResults(
        observed_limit,
        expected_limit,
        observed_CLs,
        expected_CLs,
        poi_values,
        confidence_level,
    )
    assert limit_results.observed_limit == observed_limit
    assert np.allclose(limit_results.expected_limit, expected_limit)
    assert np.allclose(limit_results.observed_CLs, observed_CLs)
    assert np.allclose(limit_results.expected_CLs, expected_CLs)
    assert np.allclose(limit_results.poi_values, poi_values)
    assert np.allclose(limit_results.confidence_level, confidence_level)


def test_SignificanceResults():
    obs_p_val = 0.02
    obs_significance = 2
    exp_p_val = 0.16
    exp_significance = 1
    significance_results = fit.SignificanceResults(
        obs_p_val, obs_significance, exp_p_val, exp_significance
    )
    assert significance_results.observed_p_value == obs_p_val
    assert significance_results.observed_significance == obs_significance
    assert significance_results.expected_p_value == exp_p_val
    assert significance_results.expected_significance == exp_significance


def test_print_results(caplog):
    caplog.set_level(logging.DEBUG)

    bestfit = np.asarray([1.0, 2.0])
    uncertainty = np.asarray([0.1, 0.3])
    labels = ["param_A", "param_B"]
    fit_results = fit.FitResults(bestfit, uncertainty, labels, np.empty(0), 0.0)

    fit.print_results(fit_results)
    assert "param_A =  1.0000 +/- 0.1000" in [rec.message for rec in caplog.records]
    assert "param_B =  2.0000 +/- 0.3000" in [rec.message for rec in caplog.records]
    caplog.clear()
