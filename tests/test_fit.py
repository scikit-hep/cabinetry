import logging

import numpy as np
import pyhf
import pytest

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


def test_print_results(caplog):
    caplog.set_level(logging.DEBUG)

    bestfit = np.asarray([1.0, 2.0])
    uncertainty = np.asarray([0.1, 0.3])
    labels = ["param_A", "param_B"]
    fit.print_results(bestfit, uncertainty, labels)
    assert "param_A:  1.000000 +/- 0.100000" in [rec.message for rec in caplog.records]
    assert "param_B:  2.000000 +/- 0.300000" in [rec.message for rec in caplog.records]
    caplog.clear()


def test_build_Asimov_data(example_spec):
    ws = pyhf.Workspace(example_spec)
    model = ws.model()
    assert np.allclose(fit.build_Asimov_data(model), [51.839756, 1])


# skip a "RuntimeWarning: numpy.ufunc size changed" warning
# due to different numpy versions used in dependencies
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fit(example_spec):
    fit_results = fit.fit(example_spec)
    assert np.allclose(fit_results.bestfit, [1.00016745, 9.16046294])
    assert np.allclose(fit_results.uncertainty, [0.04953864, 0.6131055])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 3.83055796)
    assert np.allclose(fit_results.corr_mat, [[1.0, -0.73346895], [-0.73346895, 1.0]])

    # Asimov fit
    fit_results = fit.fit(example_spec, asimov=True)
    assert np.allclose(fit_results.bestfit, [1.0, 1.0], rtol=1e-4)
    assert np.allclose(fit_results.uncertainty, [0.04956413, 0.1474005])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 1.61824936)
    assert np.allclose(fit_results.corr_mat, [[1.0, -0.33610697], [-0.33610697, 1.0]])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_custom_fit(example_spec):
    fit_results = fit.custom_fit(example_spec)
    # compared to fit(), the gamma is fixed
    assert np.allclose(fit_results.bestfit, [1.1, 8.32985794])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.38153392])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 7.90080378)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # Asimov fit, with fixed gamma (fixed not to Asimov MLE)
    fit_results = fit.custom_fit(example_spec, asimov=True)
    # the gamma factor is multiplicative and fixed to 1.1, so the
    # signal strength needs to be 1/1.1 to compensate
    assert np.allclose(fit_results.bestfit, [1.1, 0.90917877])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.12623172])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 5.68851093)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])
