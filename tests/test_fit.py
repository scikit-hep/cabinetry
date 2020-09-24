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
def test__fit_model_custom(example_spec, example_spec_multibin):
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

    # parameters held constant via keyword argument
    model, data = model_utils.model_and_data(example_spec_multibin)
    init_pars = model.config.suggested_init()
    init_pars[0] = 0.9
    init_pars[1] = 1.1
    fix_pars = model.config.suggested_fixed()
    fix_pars[0] = True
    fix_pars[1] = True
    fit_results = fit._fit_model_custom(
        model, data, init_pars=init_pars, fix_pars=fix_pars
    )
    assert np.allclose(fit_results.bestfit, [0.9, 1.1, 1.48041923, 0.97511112])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.0, 0.20694409, 0.11792805])
    assert np.allclose(fit_results.best_twice_nll, 10.45318909)


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


@mock.patch(
    "cabinetry.fit._fit_model_custom",
    side_effect=[
        fit.FitResults(
            np.asarray([0.9, 1.3]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 0.7]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 1.2]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 0.8]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        # for second fit with fixed parameter
        fit.FitResults(
            np.asarray([0.9, 1.2]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 0.8]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
    ],
)
def test_ranking(mock_fit, example_spec):
    example_spec["measurements"][0]["config"]["parameters"][0]["fixed"] = False
    bestfit = np.asarray([0.9, 1.0])
    uncertainty = np.asarray([0.02, 0.1])
    labels = ["staterror", "mu"]
    fit_results = fit.FitResults(bestfit, uncertainty, labels, np.empty(0), 0.0)
    ranking_results = fit.ranking(example_spec, fit_results)

    # correct call to fit
    expected_fix = [True, False]
    expected_inits = [[0.94956657, 1.0], [0.85043343, 1.0], [0.92, 1.0], [0.88, 1.0]]
    assert mock_fit.call_count == 4
    for i in range(4):
        assert np.allclose(
            mock_fit.call_args_list[i][1]["init_pars"], expected_inits[i]
        )
        assert np.allclose(mock_fit.call_args_list[i][1]["fix_pars"], expected_fix)

    # POI removed from fit results
    assert np.allclose(ranking_results.bestfit, [0.9])
    assert np.allclose(ranking_results.uncertainty, [0.02])
    assert ranking_results.labels == ["staterror"]

    # received correct mock results
    assert np.allclose(ranking_results.prefit_up, [0.3])
    assert np.allclose(ranking_results.prefit_down, [-0.3])
    assert np.allclose(ranking_results.postfit_up, [0.2])
    assert np.allclose(ranking_results.postfit_down, [-0.2])

    # fixed parameter in ranking
    example_spec["measurements"][0]["config"]["parameters"][0]["fixed"] = True
    ranking_results = fit.ranking(example_spec, fit_results)
    # expect two calls in this ranking (and had 4 before, so 6 total): pre-fit
    # uncertainty is 0 since parameter is fixed, mock post-fit uncertainty is not 0
    assert mock_fit.call_count == 6
    assert np.allclose(ranking_results.prefit_up, [0.0])
    assert np.allclose(ranking_results.prefit_down, [0.0])
    assert np.allclose(ranking_results.postfit_up, [0.2])
    assert np.allclose(ranking_results.postfit_down, [-0.2])


@mock.patch(
    "cabinetry.fit._fit_model_custom",
    side_effect=[
        fit.FitResults(
            np.asarray([0.9, 1.3]), np.asarray([0.1, 0.1]), [], np.empty(0), 8.0
        )
    ]  # nominal fit
    + [
        fit.FitResults(np.empty(0), np.empty(0), [], np.empty(0), abs(i) + 8)
        for i in np.linspace(-5, 5, 11)
    ]  # fits in scan
    + [
        fit.FitResults(
            np.asarray([0.9, 1.3]), np.asarray([0.1, 0.1]), [], np.empty(0), 2.0
        )
    ]
    * 6,  # fits for custom parameter range
)
def test_scan(mock_fit, example_spec):
    expected_scan_values = np.linspace(1.1, 1.5, 11)
    # -2 log(L) from unconstrained fit subtracted from expected NLLs
    expected_delta_nlls = np.abs(np.linspace(-5, 5, 11))

    par_name = "Signal strength"
    scan_results = fit.scan(example_spec, par_name)
    assert scan_results.name == par_name
    assert scan_results.bestfit == 1.3
    assert scan_results.uncertainty == 0.1
    assert np.allclose(scan_results.scanned_values, expected_scan_values)
    assert np.allclose(scan_results.delta_nlls, expected_delta_nlls)

    assert mock_fit.call_count == 12
    # unconstrained fit
    assert mock_fit.call_args_list[0][1] == {}
    # fits in scan
    for i, scan_val in enumerate(expected_scan_values):
        assert mock_fit.call_args_list[i + 1][1]["init_pars"] == [1.1, scan_val]
        assert mock_fit.call_args_list[i + 1][1]["fix_pars"] == [True, True]

    # parameter range specified
    scan_results = fit.scan(example_spec, par_name, par_range=(1.0, 1.5), n_steps=5)
    expected_custom_scan = np.linspace(1.0, 1.5, 5)
    assert np.allclose(scan_results.scanned_values, expected_custom_scan)

    # unknown parameter
    with pytest.raises(ValueError, match="could not find parameter abc in model"):
        fit.scan(example_spec, "abc")
