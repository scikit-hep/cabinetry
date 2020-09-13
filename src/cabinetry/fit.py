import logging
from typing import Any, Dict, List, NamedTuple, Optional

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


class RankingResults(NamedTuple):
    """Collects nuisance parameter ranking results in one object.

    The best-fit results per parameter, the uncertainties, and the labels should
    not include the parameter of interest, since no impact for it is calculated.

    Args:
        bestfit (numpy.ndarray): best-fit results of parameters
        uncertainty (numpy.ndarray): uncertainties of best-fit parameter results
        labels (List[str]): parameter labels
        prefit_up (numpy.ndarray): pre-fit impact in "up" direction
        prefit_down (numpy.ndarray): pre-fit impact in "down" direction
        postfit_up (numpy.ndarray): post-fit impact in "up" direction
        postfit_down (numpy.ndarray): post-fit impact in "down" direction
    """

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List[str]
    prefit_up: np.ndarray
    prefit_down: np.ndarray
    postfit_up: np.ndarray
    postfit_down: np.ndarray


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


def _fit_model_custom(
    model: pyhf.pdf.Model,
    data: List[float],
    fixed_pars: Optional[List[Dict[str, Any]]] = None,
) -> FitResults:
    """Uses ``iminuit`` directly to perform a maximum likelihood fit.

    Parameters set to be fixed in the model are held constant.

    Args:
        model (pyhf.pdf.Model): the model to use in the fit
        data (List[float]): the data to fit the model to
        fixed_pars: Optional[List[Dict[str, Any]]]: list of parameters to
            hold constant at given value, defaults to None (no additional
            parameters held constant)

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fix_pars = model.config.suggested_fixed()

    labels = model_utils.get_parameter_names(model)

    if fixed_pars:
        # hold parameters parameters
        for par in fixed_pars:
            par_index = par["index"]
            par_value = par["value"]
            log.debug(f"holding parameter {labels[par_index]} fixed at {par_value:.6f}")
            fix_pars[par_index] = True
            init_pars[par_index] = par_value

    # set initial step size to 0 for fixed parameters
    # this will cause the associated parameter uncertainties to be 0 post-fit
    step_size = [0.1 if not fix_pars[i_par] else 0.0 for i_par in range(len(init_pars))]

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

    model, data = model_utils.model_and_data(spec, asimov=asimov)

    if not custom:
        fit_result = _fit_model_pyhf(model, data)
    else:
        fit_result = _fit_model_custom(model, data)

    print_results(fit_result)
    log.debug(f"-2 log(L) = {fit_result.best_twice_nll:.6f} at the best-fit point")

    return fit_result


def ranking(
    spec: Dict[str, Any], fit_results: FitResults, asimov: bool = False
) -> RankingResults:
    """Calculates the impact of nuisance parameters on the parameter of interest (POI).

    The impact is given by the difference in the POI between the nominal fit, and a fit
    where the nuisance parameter is held constant at its nominal value plus/minus its
    associated uncertainty. The ``asimov`` flag determines which dataset is used to
    calculate the impact. To calculate the proper Asimov impact, the ``fit_results``
    should come from a fit to the Asimov dataset. The "pre-fit impact" is obtained by
    varying the nuisance parameters by their uncertainty given by their constraint term.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        fit_results (FitResults): fit results to use for ranking
        asimov (bool, optional): whether to fit the Asimov dataset, defaults
            to False

    Returns:
        RankingResults: fit results for parameters, and pre- and post-fit impacts
    """
    model, data = model_utils.model_and_data(spec, asimov=asimov)
    labels = model_utils.get_parameter_names(model)
    prefit_unc = model_utils.get_prefit_uncertainties(model)
    nominal_poi = fit_results.bestfit[model.config.poi_index]

    all_impacts = []
    for i_par, label in enumerate(labels):
        if label == model.config.poi_name:
            continue  # do not calculate impact of POI on itself
        log.info(f"running ranking for {label}")

        parameter_impacts = []
        # calculate impacts: pre-fit up, pre-fit down, post-fit up, post-fit down
        for val in [
            fit_results.bestfit[i_par] + prefit_unc[i_par],
            fit_results.bestfit[i_par] - prefit_unc[i_par],
            fit_results.bestfit[i_par] + fit_results.uncertainty[i_par],
            fit_results.bestfit[i_par] - fit_results.uncertainty[i_par],
        ]:
            # could skip pre-fit calculation for unconstrained parameters
            fix_dict = [{"index": i_par, "value": val}]
            fit_results_ranking = _fit_model_custom(model, data, fixed_pars=fix_dict)
            poi_val = fit_results_ranking.bestfit[model.config.poi_index]
            parameter_impact = poi_val - nominal_poi
            log.info(
                f"POI is {poi_val:.6f}, difference to nominal is {parameter_impact:.6f}"
            )
            parameter_impacts.append(parameter_impact)
        all_impacts.append(parameter_impacts)

    all_impacts_np = np.asarray(all_impacts)
    prefit_up = all_impacts_np[:, 0]
    prefit_down = all_impacts_np[:, 1]
    postfit_up = all_impacts_np[:, 2]
    postfit_down = all_impacts_np[:, 3]

    # remove parameter of interest from bestfit / uncertainty / labels
    # such that their entries match the entries of the impacts
    bestfit = np.delete(fit_results.bestfit, model.config.poi_index)
    uncertainty = np.delete(fit_results.uncertainty, model.config.poi_index)
    labels = np.delete(fit_results.labels, model.config.poi_index).tolist()

    ranking_results = RankingResults(
        bestfit, uncertainty, labels, prefit_up, prefit_down, postfit_up, postfit_down
    )
    return ranking_results
