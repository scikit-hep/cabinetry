import logging
from typing import Any, Dict, List, NamedTuple, Tuple

import iminuit
import numpy as np
import pyhf

from . import model_utils


log = logging.getLogger(__name__)


class FitResults(NamedTuple):
    """Collects fit results in one object.

    Args:
        bestfit (numpy.ndarray): best-fit results of parameters
        uncertainty (numpy.ndarray): uncertainties of best-fit parameter results
        labels (List[str]): parameter labels
        corr_mat (np.ndarray): parameter correlation matrix
        best_twice_nll (float): -2 log(likelihood) at best-fit point
    """

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List[str]
    corr_mat: np.ndarray
    best_twice_nll: float


def print_results(
    fit_result: FitResults,
) -> None:
    """Prints the best-fit parameter results and associated uncertainties.

    Args:
        fit_result (FitResults): results of fit to be printed
    """
    max_label_length = max([len(label) for label in fit_result.labels])
    for i, label in enumerate(fit_result.labels):
        log.info(
            f"{label.ljust(max_label_length)}: {fit_result.bestfit[i]: .6f} +/- {fit_result.uncertainty[i]:.6f}"
        )


def model_and_data(
    spec: Dict[str, Any], asimov: bool = False
) -> Tuple[pyhf.pdf.Model, List[float]]:
    """Returns model and data for a ``pyhf`` workspace specification.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to return the Asimov dataset, defaults
            to False

    Returns:
        Tuple[pyhf.pdf.Model, List[float]]:
            - a HistFactory-style model in ``pyhf`` format
            - the data (plus auxdata) for the model
    """
    workspace = pyhf.Workspace(spec)
    model = workspace.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        }
    )  # use HistFactory InterpCode=4
    if not asimov:
        data = workspace.data(model)
    else:
        data = model_utils.build_Asimov_data(model)
    return model, data


def _fit_model_pyhf(model: pyhf.pdf.Model, data: List[float]) -> FitResults:
    """Uses the ``pyhf.infer`` API to perform a maximum likelihood fit.

    Parameters set to be fixed in the model are held constant.

    Args:
        model (pyhf.pdf.Model): the model to use in the fit
        data (List[float]): the data to fit the model to

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    result, result_obj = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_result_obj=True
    )

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = model_utils.get_parameter_names(model)
    corr_mat = result_obj.minuit.np_matrix(correlation=True, skip_fixed=False)
    best_twice_nll = float(result_obj.fun)  # convert 0-dim np.ndarray to float

    fit_result = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)
    return fit_result


def _fit_model_custom(model: pyhf.pdf.Model, data: List[float]) -> FitResults:
    """Uses ``iminuit`` directly to perform a maximum likelihood fit.

    Parameters set to be fixed in the model are held constant.

    Args:
        model (pyhf.pdf.Model): the model to use in the fit
        data (List[float]): the data to fit the model to

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fix_pars = model.config.suggested_fixed()

    # set initial step size to 0 for fixed parameters
    # this will cause the associated parameter uncertainties to be 0 post-fit
    step_size = [0.1 if not fix_pars[i_par] else 0.0 for i_par in range(len(init_pars))]

    labels = model_utils.get_parameter_names(model)

    def twice_nll_func(pars: np.ndarray) -> np.float64:
        twice_nll = -2 * model.logpdf(pars, data)
        return twice_nll[0]

    m = iminuit.Minuit.from_array_func(
        twice_nll_func,
        init_pars,
        error=step_size,
        limit=par_bounds,
        fix=fix_pars,
        name=labels,
        errordef=1,
        print_level=1,
    )
    # decrease tolerance (goal: EDM < 0.002*tol*errordef), default tolerance is 0.1
    m.tol /= 10
    m.migrad()
    m.hesse()

    bestfit = m.np_values()
    uncertainty = m.np_errors()
    corr_mat = m.np_matrix(correlation=True, skip_fixed=False)
    best_twice_nll = m.fval

    fit_result = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)
    return fit_result


def fit(spec: Dict[str, Any], asimov: bool = False, custom: bool = False) -> FitResults:
    """Performs a  maximum likelihood fit, reports and returns the results.

    The ``asimov`` flag allows to fit the Asimov dataset instead of observed
    data. Depending on the ``custom`` keyword argument, this uses either the
    ``pyhf.infer`` API or ``iminuit`` directly for more control over the
    minimization.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to fit the Asimov dataset, defaults
            to False
        custom (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

    Returns:
        FitResults: object storing relevant fit results
    """
    log.info("performing maximum likelihood fit")

    model, data = model_and_data(spec, asimov)

    if not custom:
        fit_result = _fit_model_pyhf(model, data)
    else:
        fit_result = _fit_model_custom(model, data)

    print_results(fit_result)
    log.debug(f"-2 log(L) = {fit_result.best_twice_nll:.6f} at the best-fit point")

    return fit_result
