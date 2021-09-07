import logging
import re
from unittest import mock

import iminuit
import numpy as np
import pytest

from cabinetry import fit
from cabinetry import model_utils


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


# skip a "RuntimeWarning: numpy.ufunc size changed" warning
# due to different numpy versions used in dependencies
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@mock.patch("cabinetry.fit._run_minos")
def test__fit_model_pyhf(mock_minos, example_spec, example_spec_multibin):
    model, data = model_utils.model_and_data(example_spec)
    fit_results = fit._fit_model_pyhf(model, data)
    assert np.allclose(fit_results.bestfit, [1.1, 8.33624084])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.38182003])
    assert fit_results.labels == ["staterror_Signal-Region[0]", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 7.82495235)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # Asimov fit, with fixed gamma (fixed not to Asimov MLE)
    model, data = model_utils.model_and_data(example_spec, asimov=True)
    fit_results = fit._fit_model_pyhf(model, data)
    # the gamma factor is multiplicative and fixed to 1.1, so the
    # signal strength needs to be 1/1.1 to compensate
    assert np.allclose(fit_results.bestfit, [1.1, 0.90917877])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.12628017])
    assert fit_results.labels == ["staterror_Signal-Region[0]", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 5.61189476)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # parameters held constant via keyword argument
    model, data = model_utils.model_and_data(example_spec_multibin)
    init_pars = model.config.suggested_init()
    init_pars[0] = 0.9
    init_pars[1] = 1.1
    fix_pars = model.config.suggested_fixed()
    fix_pars[0] = True
    fix_pars[1] = True
    fit_results = fit._fit_model_pyhf(
        model, data, init_pars=init_pars, fix_pars=fix_pars
    )
    assert np.allclose(fit_results.bestfit, [0.9, 1.1, 1.48041923, 0.97511112])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.0, 0.20694409, 0.11792805])
    assert np.allclose(fit_results.best_twice_nll, 10.4531891)

    # including minos, one parameter is unknown
    model, data = model_utils.model_and_data(example_spec)
    fit_results = fit._fit_model_pyhf(model, data, minos=["Signal strength", "abc"])
    assert mock_minos.call_count == 1
    # first argument to minos call is the Minuit instance
    assert mock_minos.call_args[0][1] == ["Signal strength", "abc"]
    assert mock_minos.call_args[0][2] == [
        "staterror_Signal-Region[0]",
        "Signal strength",
    ]
    assert mock_minos.call_args[1] == {}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@mock.patch("cabinetry.fit._run_minos")
def test__fit_model_custom(mock_minos, example_spec, example_spec_multibin):
    model, data = model_utils.model_and_data(example_spec)
    fit_results = fit._fit_model_custom(model, data)
    assert np.allclose(fit_results.bestfit, [1.1, 8.33625071])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.38182151])
    assert fit_results.labels == ["staterror_Signal-Region[0]", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 7.82495235)
    assert np.allclose(fit_results.corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # Asimov fit, with fixed gamma (fixed not to Asimov MLE)
    model, data = model_utils.model_and_data(example_spec, asimov=True)
    fit_results = fit._fit_model_custom(model, data)
    # the gamma factor is multiplicative and fixed to 1.1, so the
    # signal strength needs to be 1/1.1 to compensate
    assert np.allclose(fit_results.bestfit, [1.1, 0.90917877])
    assert np.allclose(fit_results.uncertainty, [0.0, 0.12628023])
    assert fit_results.labels == ["staterror_Signal-Region[0]", "Signal strength"]
    assert np.allclose(fit_results.best_twice_nll, 5.61189476)
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

    # including minos
    model, data = model_utils.model_and_data(example_spec)
    fit_results = fit._fit_model_custom(model, data, minos=["Signal strength"])
    assert mock_minos.call_count == 1
    # first argument to minos call is the Minuit instance
    assert mock_minos.call_args[0][1] == ["Signal strength"]
    assert mock_minos.call_args[0][2] == [
        "staterror_Signal-Region[0]",
        "Signal strength",
    ]
    assert mock_minos.call_args[1] == {}


@mock.patch(
    "cabinetry.fit._fit_model_custom",
    return_value=fit.FitResults(
        np.asarray([1.2]), np.asarray([0.2]), ["par"], np.empty(0), 2.0
    ),
)
@mock.patch(
    "cabinetry.fit._fit_model_pyhf",
    return_value=fit.FitResults(
        np.asarray([1.1]), np.asarray([0.2]), ["par"], np.empty(0), 2.0
    ),
)
def test__fit_model(mock_pyhf, mock_custom, example_spec):
    model, data = model_utils.model_and_data(example_spec)

    # pyhf API
    fit_results = fit._fit_model(model, data)
    assert mock_pyhf.call_count == 1
    assert mock_pyhf.call_args[0][0].spec == model.spec
    assert mock_pyhf.call_args[0][1] == data
    assert mock_pyhf.call_args[1] == {
        "init_pars": None,
        "fix_pars": None,
        "minos": None,
    }
    assert np.allclose(fit_results.bestfit, [1.1])

    # pyhf API, init/fixed pars, minos
    fit_results = fit._fit_model(
        model,
        data,
        init_pars=[1.5, 2.0],
        fix_pars=[False, True],
        minos=["Signal strength"],
    )
    assert mock_pyhf.call_count == 2
    assert mock_pyhf.call_args[0][0].spec == model.spec
    assert mock_pyhf.call_args[0][1] == data
    assert mock_pyhf.call_args[1] == {
        "init_pars": [1.5, 2.0],
        "fix_pars": [False, True],
        "minos": ["Signal strength"],
    }
    assert np.allclose(fit_results.bestfit, [1.1])

    # direct iminuit
    fit_results = fit._fit_model(model, data, custom_fit=True)
    assert mock_custom.call_count == 1
    assert mock_custom.call_args[0][0].spec == model.spec
    assert mock_custom.call_args[0][1] == data
    assert mock_custom.call_args[1] == {
        "init_pars": None,
        "fix_pars": None,
        "minos": None,
    }
    assert np.allclose(fit_results.bestfit, [1.2])

    # direct iminuit, init/fixed pars, minos
    fit_results = fit._fit_model(
        model,
        data,
        init_pars=[1.5, 2.0],
        fix_pars=[False, True],
        minos=["Signal strength"],
        custom_fit=True,
    )
    assert mock_custom.call_count == 2
    assert mock_custom.call_args[0][0].spec == model.spec
    assert mock_custom.call_args[0][1] == data
    assert mock_custom.call_args[1] == {
        "init_pars": [1.5, 2.0],
        "fix_pars": [False, True],
        "minos": ["Signal strength"],
    }
    assert np.allclose(fit_results.bestfit, [1.2])


def test__run_minos(caplog):
    caplog.set_level(logging.DEBUG)

    def func_to_minimize(pars):
        # mock NLL
        return (
            np.sum(
                np.power(pars - 2 * np.ones_like(pars), 2)
                + np.power(pars - 1 * np.ones_like(pars), 4)
            )
            + pars[0]
        )

    m = iminuit.Minuit(func_to_minimize, [1.0, 1.0], name=["a", "b"])
    m.errordef = 1
    m.migrad()
    fit._run_minos(m, ["b"], ["a", "b"])
    assert "running MINOS for b" in [rec.message for rec in caplog.records]
    assert "b =  1.5909 -0.7262 +0.4738" in [rec.message for rec in caplog.records]
    caplog.clear()

    # unknown parameter, MINOS does not run
    m = iminuit.Minuit(func_to_minimize, [1.0, 1.0])
    m.errordef = 1
    m.migrad()
    fit._run_minos(m, ["x2"], ["a", "b"])
    assert [rec.message for rec in caplog.records] == [
        "parameter x2 not found in model",
        "MINOS results:",
    ]
    caplog.clear()


@mock.patch("cabinetry.model_utils.unconstrained_parameter_count", return_value=1)
def test__goodness_of_fit(
    mock_count, example_spec_multibin, example_spec_no_aux, caplog
):
    caplog.set_level(logging.DEBUG)

    model, data = model_utils.model_and_data(example_spec_multibin)
    p_val = fit._goodness_of_fit(model, data, 9.964913)
    assert mock_count.call_count == 1
    assert mock_count.call_args[0][0].spec == model.spec
    assert mock_count.call_args[1] == {}
    assert "Delta NLL = 0.084185" in [rec.message for rec in caplog.records]
    assert np.allclose(p_val, 0.91926079)
    caplog.clear()

    # no auxdata and zero degrees of freedom in chi2 test
    model, data = model_utils.model_and_data(example_spec_no_aux)
    p_val = fit._goodness_of_fit(model, data, 6.01482863)
    assert mock_count.call_count == 2
    assert mock_count.call_args[0][0].spec == model.spec
    assert mock_count.call_args[1] == {}
    assert (
        "cannot calculate p-value: 0 degrees of freedom and Delta NLL = 0.000000"
        in [rec.message for rec in caplog.records]
    )
    assert np.isnan(p_val)
    caplog.clear()


@mock.patch("cabinetry.fit._goodness_of_fit", return_value=0.1)
@mock.patch("cabinetry.fit.print_results")
@mock.patch(
    "cabinetry.fit._fit_model",
    return_value=fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["par"], np.empty(0), 2.0
    ),
)
def test_fit(mock_fit, mock_print, mock_gof):
    model = mock.MagicMock()
    data = mock.MagicMock()

    # fit through pyhf.infer API
    fit_results = fit.fit(model, data)
    assert mock_fit.call_args_list == [
        ((model, data), {"minos": None, "custom_fit": False})
    ]
    mock_print.assert_called_once()
    assert mock_print.call_args[0][0].bestfit == [1.0]
    assert mock_print.call_args[0][0].uncertainty == [0.1]
    assert mock_print.call_args[0][0].labels == ["par"]
    assert fit_results.bestfit == [1.0]

    # custom fit
    fit_results = fit.fit(model, data, custom_fit=True)
    assert mock_fit.call_count == 2
    assert mock_fit.call_args == ((model, data), {"minos": None, "custom_fit": True})
    assert mock_print.call_args[0][0].bestfit == [1.0]
    assert mock_print.call_args[0][0].uncertainty == [0.1]
    assert fit_results.bestfit == [1.0]

    # parameters for MINOS
    fit_results = fit.fit(model, data, minos=["abc"])
    assert mock_fit.call_count == 3
    assert mock_fit.call_args[1] == {"minos": ["abc"], "custom_fit": False}
    assert fit_results.bestfit == [1.0]
    fit_results = fit.fit(model, data, minos="abc", custom_fit=True)
    assert mock_fit.call_count == 4
    assert mock_fit.call_args[1] == {"minos": ["abc"], "custom_fit": True}
    assert fit_results.bestfit == [1.0]

    # goodness-of-fit test
    fit_results_gof = fit.fit(model, data, goodness_of_fit=True)
    assert mock_gof.call_args_list == [((model, data, 2.0), {})]
    assert fit_results_gof.goodness_of_fit == 0.1


@mock.patch(
    "cabinetry.fit._fit_model",
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
        # for second ranking call with fixed parameter
        fit.FitResults(
            np.asarray([0.9, 1.2]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 0.8]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        # for third ranking call without reference results
        fit.FitResults(
            np.asarray([0.9, 1.0]), np.asarray([0.3, 0.3]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 1.3]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
        fit.FitResults(
            np.asarray([0.9, 0.7]), np.asarray([0.1, 0.1]), ["a", "b"], np.empty(0), 0.0
        ),
    ],
)
def test_ranking(mock_fit, example_spec):
    example_spec["measurements"][0]["config"]["parameters"][0]["fixed"] = False
    bestfit = np.asarray([0.9, 1.0])
    uncertainty = np.asarray([0.02, 0.1])
    labels = ["staterror", "mu"]
    fit_results = fit.FitResults(bestfit, uncertainty, labels, np.empty(0), 0.0)
    model, data = model_utils.model_and_data(example_spec)
    ranking_results = fit.ranking(model, data, fit_results=fit_results)

    # correct call to fit
    expected_fix = [True, False]
    expected_inits = [[0.95019305, 1.0], [0.84980695, 1.0], [0.92, 1.0], [0.88, 1.0]]
    assert mock_fit.call_count == 4
    for i in range(4):
        assert mock_fit.call_args_list[i][0] == (model, data)
        assert np.allclose(
            mock_fit.call_args_list[i][1]["init_pars"], expected_inits[i]
        )
        assert np.allclose(mock_fit.call_args_list[i][1]["fix_pars"], expected_fix)
        assert mock_fit.call_args_list[i][1]["custom_fit"] is False

    # POI removed from fit results
    assert np.allclose(ranking_results.bestfit, [0.9])
    assert np.allclose(ranking_results.uncertainty, [0.02])
    assert ranking_results.labels == ["staterror"]

    # received correct mock results
    assert np.allclose(ranking_results.prefit_up, [0.3])
    assert np.allclose(ranking_results.prefit_down, [-0.3])
    assert np.allclose(ranking_results.postfit_up, [0.2])
    assert np.allclose(ranking_results.postfit_down, [-0.2])

    # fixed parameter in ranking, custom fit
    example_spec["measurements"][0]["config"]["parameters"][0]["fixed"] = True
    model, data = model_utils.model_and_data(example_spec)
    ranking_results = fit.ranking(model, data, fit_results=fit_results, custom_fit=True)
    # expect two calls in this ranking (and had 4 before, so 6 total): pre-fit
    # uncertainty is 0 since parameter is fixed, mock post-fit uncertainty is not 0
    assert mock_fit.call_count == 6
    assert mock_fit.call_args[1]["custom_fit"] is True
    assert np.allclose(ranking_results.prefit_up, [0.0])
    assert np.allclose(ranking_results.prefit_down, [0.0])
    assert np.allclose(ranking_results.postfit_up, [0.2])
    assert np.allclose(ranking_results.postfit_down, [-0.2])

    # no reference results
    ranking_results = fit.ranking(model, data, custom_fit=True)
    assert mock_fit.call_count == 9
    # reference fit
    assert mock_fit.call_args_list[-3] == ((model, data), {"custom_fit": True})
    # fits for impact
    assert mock_fit.call_args_list[-2][0] == (model, data)
    assert np.allclose(mock_fit.call_args_list[-2][1]["init_pars"], [1.2, 1.0])
    assert mock_fit.call_args_list[-2][1]["fix_pars"] == [True, False]
    assert mock_fit.call_args_list[-2][1]["custom_fit"] is True
    assert mock_fit.call_args_list[-1][0] == (model, data)
    assert np.allclose(mock_fit.call_args_list[-1][1]["init_pars"], [0.6, 1.0])
    assert mock_fit.call_args_list[-1][1]["fix_pars"] == [True, False]
    assert mock_fit.call_args_list[-1][1]["custom_fit"] is True
    # ranking results
    assert np.allclose(ranking_results.prefit_up, [0.0])
    assert np.allclose(ranking_results.prefit_down, [0.0])
    assert np.allclose(ranking_results.postfit_up, [0.3])
    assert np.allclose(ranking_results.postfit_down, [-0.3])


@mock.patch(
    "cabinetry.fit._fit_model",
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
    model, data = model_utils.model_and_data(example_spec)

    par_name = "Signal strength"
    scan_results = fit.scan(model, data, par_name)
    assert scan_results.name == par_name
    assert scan_results.bestfit == 1.3
    assert scan_results.uncertainty == 0.1
    assert np.allclose(scan_results.parameter_values, expected_scan_values)
    assert np.allclose(scan_results.delta_nlls, expected_delta_nlls)

    assert mock_fit.call_count == 12
    # unconstrained fit
    assert mock_fit.call_args_list[0][0] == ((model, data))
    assert mock_fit.call_args_list[0][1] == {"custom_fit": False}
    # fits in scan
    for i, scan_val in enumerate(expected_scan_values):
        assert mock_fit.call_args_list[i + 1][0] == ((model, data))
        assert mock_fit.call_args_list[i + 1][1]["init_pars"] == [1.1, scan_val]
        assert mock_fit.call_args_list[i + 1][1]["fix_pars"] == [True, True]
        assert mock_fit.call_args_list[i + 1][1]["custom_fit"] is False

    # parameter range specified, custom fit
    scan_results = fit.scan(
        model, data, par_name, par_range=(1.0, 1.5), n_steps=5, custom_fit=True
    )
    expected_custom_scan = np.linspace(1.0, 1.5, 5)
    assert np.allclose(scan_results.parameter_values, expected_custom_scan)
    assert mock_fit.call_args[1]["custom_fit"] is True

    # unknown parameter
    with pytest.raises(ValueError, match="parameter abc not found in model"):
        fit.scan(model, data, "abc")


def test_limit(example_spec_with_background, caplog):
    caplog.set_level(logging.DEBUG)

    # expected values for results
    observed_limit = 0.749
    expected_limit = [0.303, 0.411, 0.581, 0.833, 1.160]

    # modify workspace to include custom POI range
    example_spec_with_background["measurements"][0]["config"]["parameters"][0][
        "bounds"
    ] = [[0, 8]]
    model, data = model_utils.model_and_data(example_spec_with_background)

    limit_results = fit.limit(model, data)
    assert np.allclose(limit_results.observed_limit, observed_limit, rtol=1e-2)
    assert np.allclose(limit_results.expected_limit, expected_limit, rtol=1e-2)
    # compare a few CLs values
    assert np.allclose(limit_results.observed_CLs[0], 0.780874)
    assert np.allclose(
        limit_results.expected_CLs[0],
        [0.402421, 0.548538, 0.719383, 0.878530, 0.971678],
    )
    assert np.allclose(limit_results.poi_values[0], 0.1)
    assert np.allclose(limit_results.observed_CLs[-1], 0.0)
    assert np.allclose(limit_results.expected_CLs[-1], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(limit_results.poi_values[-1], 8.0)  # from custom POI range
    # verify that POI values are sorted
    assert np.allclose(limit_results.poi_values, sorted(limit_results.poi_values))
    caplog.clear()

    # access negative POI values with lower bracket below zero
    limit_results = fit.limit(model, data, bracket=(-1, 5), tolerance=0.05)
    assert "skipping fit for Signal strength = -1.0000, setting CLs = 1" in [
        rec.message for rec in caplog.records
    ]
    assert np.allclose(limit_results.observed_limit, observed_limit, rtol=5e-2)
    assert np.allclose(limit_results.expected_limit, expected_limit, rtol=5e-2)
    caplog.clear()

    # convergence issues due to number of iterations
    fit.limit(model, data, bracket=(0.1, 1), maxiter=1)
    assert "one or more calculations did not converge, check log" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # Asimov dataset with nominal signal strength of 0
    example_spec_with_background["measurements"][0]["config"]["parameters"][0][
        "inits"
    ] = [0.0]
    model, data = model_utils.model_and_data(example_spec_with_background, asimov=True)
    limit_results = fit.limit(model, data)
    assert np.allclose(limit_results.observed_limit, 0.586, rtol=2e-2)
    assert np.allclose(limit_results.expected_limit, expected_limit, rtol=2e-2)
    caplog.clear()

    # bracket does not contain root
    with pytest.raises(
        ValueError, match=re.escape("f(a) and f(b) must have different signs")
    ):
        fit.limit(model, data, bracket=(1.0, 2.0))
        assert (
            "CLs values at 1.000 and 2.000 do not bracket CLs=0.05, try a different "
            "starting bracket" in [rec.message for rec in caplog.records]
        )
    caplog.clear()

    # bracket with identical values
    with pytest.raises(ValueError, match="the two bracket values must not be the same"):
        fit.limit(model, data, bracket=(3.0, 3.0))
    caplog.clear()


def test_significance(example_spec_with_background):
    # increase observed data for smaller observed p-value
    example_spec_with_background["observations"][0]["data"] = [196]

    model, data = model_utils.model_and_data(example_spec_with_background)
    significance_results = fit.significance(model, data)
    assert np.allclose(significance_results.observed_p_value, 0.00080517)
    assert np.allclose(significance_results.observed_significance, 3.15402672)
    assert np.allclose(significance_results.expected_p_value, 0.00033333)
    assert np.allclose(significance_results.expected_significance, 3.40293444)

    # reduce signal for larger expected p-value
    example_spec_with_background["channels"][0]["samples"][0]["data"] = [30]

    # Asimov dataset, observed = expected
    model, data = model_utils.model_and_data(example_spec_with_background, asimov=True)
    significance_results = fit.significance(model, data)
    assert np.allclose(significance_results.observed_p_value, 0.02062714)
    assert np.allclose(significance_results.observed_significance, 2.04096523)
    assert np.allclose(significance_results.expected_p_value, 0.02062714)
    assert np.allclose(significance_results.expected_significance, 2.04096523)
