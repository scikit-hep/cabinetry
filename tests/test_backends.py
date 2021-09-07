import numpy as np
import pyhf
import pytest

import cabinetry


@pytest.fixture
def reset_backend():
    """This sets the ``pyhf`` backend back to ``numpy`` after a test has run."""
    # setup before test
    yield  # test
    # teardown after test
    pyhf.set_backend("numpy")


@pytest.mark.slow
@pytest.mark.no_cover
@pytest.mark.parametrize("backend", ["jax", "pytorch", "tensorflow"])
def test_backend_integration(backend, reset_backend):
    """Integration test for the inference pipeline that can be run with all ``pyhf``
    backends to ensure they work. ``typeguard`` will catch type issues at runtime.
    """
    pyhf.set_backend(backend)
    # construct a simple workspace
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[24.0, 22.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    ws = pyhf.Workspace.build(model, [74, 74])

    model, data = cabinetry.model_utils.model_and_data(ws)

    # MLE comparison of both implementations
    fit_results_1 = cabinetry.fit.fit(model, data, minos="mu", goodness_of_fit=True)
    fit_results_2 = cabinetry.fit.fit(
        model, data, minos="mu", goodness_of_fit=True, custom_fit=True
    )
    assert np.allclose(fit_results_1.bestfit, fit_results_2.bestfit, rtol=0.01)
    assert np.allclose(fit_results_1.uncertainty, fit_results_2.uncertainty, rtol=0.01)
    assert np.allclose(fit_results_1.corr_mat, fit_results_2.corr_mat, rtol=0.01)
    assert np.allclose(fit_results_1.best_twice_nll, fit_results_2.best_twice_nll)
    assert np.allclose(fit_results_1.goodness_of_fit, fit_results_2.goodness_of_fit)

    # remaining types of fit results
    cabinetry.fit.ranking(model, data)
    cabinetry.fit.scan(model, data, "mu")
    cabinetry.fit.limit(model, data)
    cabinetry.fit.significance(model, data)

    # model_utils functions that deal with expected_data
    cabinetry.model_utils.asimov_data(model)
    # parameters, uncertainty, correlation for stdev
    param_values = cabinetry.model_utils.asimov_parameters(model)
    param_uncertainty = cabinetry.model_utils.prefit_uncertainties(model)
    corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
    np.fill_diagonal(corr_mat, 1.0)
    cabinetry.model_utils.yield_stdev(model, param_values, param_uncertainty, corr_mat)


def test_backend_reset():
    # ensure that the pyhf backend is numpy again
    assert pyhf.tensorlib.name == "numpy"
