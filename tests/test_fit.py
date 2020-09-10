import logging
from unittest import mock

import numpy as np
import pytest

from cabinetry import fit
from cabinetry import model_utils


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
    fit_results = fit.FitResults(bestfit, uncertainty, labels, np.empty(0), 0.0)

    fit.print_results(fit_results)
    assert "param_A:  1.000000 +/- 0.100000" in [rec.message for rec in caplog.records]
    assert "param_B:  2.000000 +/- 0.300000" in [rec.message for rec in caplog.records]
    caplog.clear()


# skip a "RuntimeWarning: numpy.ufunc size changed" warning
# due to different numpy versions used in dependencies
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test__fit_model_pyhf(example_spec):
    model, data = model_utils.model_and_data(example_spec)
    fit_results = fit._fit_model_pyhf(model, data)
    assert np.allclose(fit_results.bestfit, [1.1, 8.32984849])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.38099445])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 7.90080379)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # Asimov fit
    model, data = model_utils.model_and_data(example_spec, asimov=True)
    fit_results = fit._fit_model_pyhf(model, data)
    assert np.allclose(fit_results.bestfit, [1.1, 0.90917877], rtol=1e-4)
    assert np.allclose(fit_results.uncertainty, [0.0, 0.12623179])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 5.68851093)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test__fit_model_custom(example_spec):
    model, data = model_utils.model_and_data(example_spec)
    fit_results = fit._fit_model_custom(model, data)
    assert np.allclose(fit_results.bestfit, [1.1, 8.32985794])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.38153392])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 7.90080378)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # Asimov fit, with fixed gamma (fixed not to Asimov MLE)
    model, data = model_utils.model_and_data(example_spec, asimov=True)
    fit_results = fit._fit_model_custom(model, data)
    # the gamma factor is multiplicative and fixed to 1.1, so the
    # signal strength needs to be 1/1.1 to compensate
    assert np.allclose(fit_results.bestfit, [1.1, 0.90917877])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.12623172])
    assert fit_results.labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 5.68851093)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])


@mock.patch("cabinetry.fit.print_results")
@mock.patch(
    "cabinetry.fit._fit_model_custom",
    return_value=fit.FitResults(
        np.asarray([3.0]), np.asarray([0.3]), ["par"], np.empty(0), 5.0
    ),
)
@mock.patch(
    "cabinetry.fit._fit_model_pyhf",
    return_value=fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["par"], np.empty(0), 2.0
    ),
)
@mock.patch("cabinetry.model_utils.model_and_data", return_value=("model", "data"))
def test_fit(mock_load, mock_pyhf, mock_custom, mock_print, example_spec):
    fit.fit(example_spec)
    assert mock_load.call_args_list == [[(example_spec,), {"asimov": False}]]
    assert mock_pyhf.call_args_list == [[("model", "data"), {}]]
    mock_print.assert_called_once()
    assert mock_print.call_args[0][0].bestfit == [1.0]
    assert mock_print.call_args[0][0].uncertainty == [0.1]
    assert mock_print.call_args[0][0].labels == ["par"]

    # Asimov fit
    fit.fit(example_spec, asimov=True)
    assert mock_load.call_args == [(example_spec,), {"asimov": True}]

    # custom fit
    fit.fit(example_spec, custom=True)
    mock_custom.assert_called_once()
    assert mock_custom.call_args == [("model", "data"), {}]
    assert mock_print.call_args[0][0].bestfit == [3.0]
    assert mock_print.call_args[0][0].uncertainty == [0.3]
