import logging
from typing import Any, Dict, List, NamedTuple

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
    bestfit: np.ndarray, uncertainty: np.ndarray, labels: List[str]
) -> None:
    """print the best-fit parameter results and associated uncertainties

    Args:
        bestfit (numpy.ndarray): best-fit results of parameters
        uncertainty (numpy.ndarray): uncertainties of best-fit parameter results
        labels (List[str]): parameter labels
    """
    max_label_length = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        l_with_spacer = label + " " * (max_label_length - len(label))
        log.info(f"{l_with_spacer}: {bestfit[i]: .6f} +/- {uncertainty[i]:.6f}")


def build_Asimov_data(model: pyhf.Model) -> np.ndarray:
    """Returns the Asimov dataset (with auxdata) for a model.

    Args:
        model (pyhf.Model): the model from which to construct the
            dataset

    Returns:
        np.ndarray: the Asimov dataset
    """
    asimov_data = np.sum(model.nominal_rates, axis=1)[0][0]
    asimov_aux = model.config.auxdata
    return np.hstack((asimov_data, asimov_aux))


def fit(spec: Dict[str, Any], asimov: bool = False) -> FitResults:
    """Performs an unconstrained maximum likelihood fit with ``pyhf``.

    Reports and returns the results of the fit. The ``asimov`` flag
    allows to fit the Asimov dataset instead of observed data.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to fit the Asimov dataset, defaults
            to False

    Returns:
        FitResults: fit information stored in one object
    """
    log.info("performing unconstrained fit")

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
        data = build_Asimov_data(model)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
    result, result_obj = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_result_obj=True
    )

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = model_utils.get_parameter_names(model)
    corr_mat = result_obj.minuit.np_matrix(correlation=True, skip_fixed=False)
    best_twice_nll = float(result_obj.fun)  # convert 0-dim np.ndarray to float

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {best_twice_nll:.6f} at the best-fit point")

    fit_result = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)
    return fit_result


def custom_fit(spec: Dict[str, Any], asimov: bool = False) -> FitResults:
    """Performs an unconstrained maximum likelihood fit with ``iminuit``.

    Reports and returns the results of the fit. The ``asimov`` flag
    allows to fit the Asimov dataset instead of observed data. Compared to
    ``fit()``, this does not use the ``pyhf.infer`` API for more control
    over the minimization.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to fit the Asimov dataset, defaults
            to False

    Returns:
        FitResults: fit information stored in one object
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    workspace = pyhf.Workspace(spec)
    model = workspace.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        }
    )  # use HistFactory InterpCode=4

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fix_pars = model.config.suggested_fixed()

    if not asimov:
        data = workspace.data(model)
    else:
        data = build_Asimov_data(model)

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

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {best_twice_nll:.6f} at the best-fit point")

    fit_result = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)
    return fit_result
