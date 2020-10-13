import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

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

    The best-fit results per parameter, the uncertainties, and the labels should not
    include the parameter of interest, since no impact for it is calculated.

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


class ScanResults(NamedTuple):
    """Collects likelihood scan results in one object.

    Args:
        name (str): name of parameter in scan
        bestfit (float): best-fit parameter value from unconstrained fit
        uncertainty (float): uncertainty of parameter in unconstrained fit
        scanned_values (np.ndarray): parameter values used in scan
        delta_nlls (np.ndarray): -2 log(L) difference at each scanned point
    """

    name: str
    bestfit: float
    uncertainty: float
    scanned_values: np.ndarray
    delta_nlls: np.ndarray


def print_results(
    fit_result: FitResults,
) -> None:
    """Prints the best-fit parameter results and associated uncertainties.

    Args:
        fit_result (FitResults): results of fit to be printed
    """
    max_label_length = max([len(label) for label in fit_result.labels])
    log.info("fit results (with symmetric uncertainties):")
    for i, label in enumerate(fit_result.labels):
        log.info(
            f"{label.ljust(max_label_length)}: {fit_result.bestfit[i]: .4f} +/- "
            f"{fit_result.uncertainty[i]:.4f}"
        )


def _fit_model_pyhf(
    model: pyhf.pdf.Model, data: List[float], minos: Optional[List[str]] = None
) -> FitResults:
    """Uses the ``pyhf.infer`` API to perform a maximum likelihood fit.

    Parameters set to be fixed in the model are held constant.

    Args:
        model (pyhf.pdf.Model): the model to use in the fit
        data (List[float]): the data to fit the model to
        minos (Optional[List[str]], optional): runs the MINOS algorithm for all
            parameters specified in the list, defaults to None (does not run MINOS)

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

    if minos is not None:
        parameters_translated = []
        for i, label in enumerate(labels):
            if label in minos:
                # pyhf does not hand over parameter names, all parameters are known as
                # x0, x1, etc.
                parameters_translated.append(f"x{i}")

        _run_minos(result_obj.minuit, parameters_translated, labels)

    return fit_result


def _fit_model_custom(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fix_pars: Optional[List[bool]] = None,
    minos: Optional[List[str]] = None,
) -> FitResults:
    """Uses ``iminuit`` directly to perform a maximum likelihood fit.

    Parameters set to be fixed in the model are held constant. The ``init_pars``
    argument allows to override the ``pyhf`` default initial parameter settings, and the
    ``fix_pars`` argument overrides which parameters are held constant.

    Args:
        model (pyhf.pdf.Model): the model to use in the fit
        data (List[float]): the data to fit the model to
        init_pars (Optional[List[float]], optional): list of initial parameter settings,
            defaults to None (use ``pyhf`` suggested inits)
        fix_pars (Optional[List[bool]], optional): list of booleans specifying which
            parameters are held constant, defaults to None (use ``pyhf`` suggestion)
        minos (Optional[List[str]], optional): runs the MINOS algorithm for all
            parameters specified in the list, defaults to None (does not run MINOS)

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    # use init_pars provided in function argument if they exist, else use default
    init_pars = init_pars or model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    # use fix_pars provided in function argument if they exist, else use default
    fix_pars = fix_pars or model.config.suggested_fixed()

    labels = model_utils.get_parameter_names(model)

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

    if minos is not None:
        _run_minos(m, minos, labels)

    return fit_result


def fit(
    spec: Dict[str, Any],
    asimov: bool = False,
    custom: bool = False,
    minos: Optional[Union[str, List[str]]] = None,
) -> FitResults:
    """Performs a  maximum likelihood fit, reports and returns the results.

    The ``asimov`` flag allows to fit the Asimov dataset instead of observed data.
    Depending on the ``custom`` keyword argument, this uses either the ``pyhf.infer``
    API or ``iminuit`` directly for more control over the minimization.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to fit the Asimov dataset, defaults to False
        custom (bool, optional): whether to use the ``pyhf.infer`` API or ``iminuit``,
            defaults to False (using ``pyhf.infer``)
        minos (Optional[Union[str, List[str]]], optional): runs the MINOS algorithm for
            all parameters specified in the list, defaults to None (does not run MINOS)

    Returns:
        FitResults: object storing relevant fit results
    """
    log.info("performing maximum likelihood fit")

    model, data = model_utils.model_and_data(spec, asimov=asimov)

    # convert minos parameter to list, if a parameter is specified and is not a list
    if minos is not None and not isinstance(minos, list):
        minos = [minos]

    if not custom:
        fit_result = _fit_model_pyhf(model, data, minos=minos)
    else:
        fit_result = _fit_model_custom(model, data, minos=minos)

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

    # get default initial parameter settings / whether parameters are constant
    init_pars_default = model.config.suggested_init()
    fix_pars_default = model.config.suggested_fixed()

    all_impacts = []
    for i_par, label in enumerate(labels):
        if label == model.config.poi_name:
            continue  # do not calculate impact of POI on itself
        log.info(f"calculating impact of {label} on {labels[model.config.poi_index]}")

        # hold current parameter constant
        fix_pars = fix_pars_default.copy()
        fix_pars[i_par] = True

        parameter_impacts = []
        # calculate impacts: pre-fit up, pre-fit down, post-fit up, post-fit down
        for np_val in [
            fit_results.bestfit[i_par] + prefit_unc[i_par],
            fit_results.bestfit[i_par] - prefit_unc[i_par],
            fit_results.bestfit[i_par] + fit_results.uncertainty[i_par],
            fit_results.bestfit[i_par] - fit_results.uncertainty[i_par],
        ]:
            # can skip pre-fit calculation for unconstrained parameters (their
            # pre-fit uncertainty is set to 0), and pre- and post-fit calculation
            # for fixed parameters (both uncertainties set to 0 as well)
            if np_val == fit_results.bestfit[i_par]:
                log.debug(f"impact of {label} is zero, skipping fit")
                parameter_impacts.append(0.0)
            else:
                init_pars = init_pars_default.copy()
                init_pars[i_par] = np_val  # set value of current nuisance parameter
                fit_results_ranking = _fit_model_custom(
                    model, data, init_pars=init_pars, fix_pars=fix_pars
                )
                poi_val = fit_results_ranking.bestfit[model.config.poi_index]
                parameter_impact = poi_val - nominal_poi
                log.debug(
                    f"POI is {poi_val:.6f}, difference to nominal is "
                    f"{parameter_impact:.6f}"
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


def scan(
    spec: Dict[str, Any],
    par_name: str,
    par_range: Optional[Tuple[float, float]] = None,
    n_steps: int = 11,
    asimov: bool = False,
) -> ScanResults:
    """Performs a likelihood scan over the specified parameter.

    If no parameter range is specified, center the scan around the best-fit result for
    the parameter that is being scanned, and scan over twice its uncertainty in each
    direction. The reported likelihood values are the differences between -2 log(L) at
    each point in the scan and the global minimum.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        par_name (str): name of parameter to scan over
        par_range (Optional[Tuple[float, float]], optional): upper and lower bounds of
            parameter in scan, defaults to None (automatically determine bounds)
        n_steps (int, optional): number of steps in scan, defaults to 10
        asimov (bool, optional): whether to fit the Asimov dataset, defaults to False

    Raises:
        ValueError: if parameter is not found in model

    Returns:
        ScanResults: includes parameter name, scanned values and 2*log(likelihood)
        offset
    """
    model, data = model_utils.model_and_data(spec, asimov=asimov)
    labels = model_utils.get_parameter_names(model)
    init_pars = model.config.suggested_init()
    fix_pars = model.config.suggested_fixed()

    # get index of parameter with name par_name
    par_index = next((i for i, label in enumerate(labels) if label == par_name), -1)
    if par_index == -1:
        raise ValueError(f"could not find parameter {par_name} in model")

    # run a fit with the parameter not held constant, to find the best-fit point
    fit_results = _fit_model_custom(model, data)
    nominal_twice_nll = fit_results.best_twice_nll
    par_mle = fit_results.bestfit[par_index]
    par_unc = fit_results.uncertainty[par_index]

    if par_range is None:
        # if no parameter range is specified, use +/-2 sigma from the MLE
        par_range = (par_mle - 2 * par_unc, par_mle + 2 * par_unc)

    scan_values = np.linspace(par_range[0], par_range[1], n_steps)
    delta_nlls = np.zeros_like(scan_values)  # holds results

    fix_pars[par_index] = True  # hold scan parameter constant in fits

    log.info(
        f"performing likelihood scan for {par_name} in range ({par_range[0]:.3f}, "
        f"{par_range[1]:.3f}) with {n_steps} steps"
    )
    for i_par, par_value in enumerate(scan_values):
        log.debug(f"performing fit with {par_name} = {par_value:.3f}")
        init_pars_scan = init_pars.copy()
        init_pars_scan[par_index] = par_value
        scan_fit_results = _fit_model_custom(
            model, data, init_pars=init_pars_scan, fix_pars=fix_pars
        )
        # subtract best-fit
        delta_nlls[i_par] = scan_fit_results.best_twice_nll - nominal_twice_nll

    scan_results = ScanResults(par_name, par_mle, par_unc, scan_values, delta_nlls)
    return scan_results


def _run_minos(
    minuit_obj: iminuit._libiminuit.Minuit, minos: List[str], labels: List[str]
) -> None:
    """Determine parameter uncertainties for a list of parameters with MINOS.

    Args:
        minuit_obj (iminuit._libiminuit.Minuit): Minuit instance to use
        minos (List[str]): parameters for which MINOS is run
        labels (List[str]]): names of all parameters known to ``iminuit``, these names
            are used in output (may be the same as the names under which ``iminiuit``
            knows parameters)
    """
    for par_name in minos:
        # get index of current parameter in labels (to translate its name if iminuit
        # did not receive the parameter labels)
        par_index = next(
            (i for i, par in enumerate(minuit_obj.parameters) if par == par_name)
        )
        log.info(f"running MINOS for {labels[par_index]}")
        minuit_obj.minos(var=par_name)

    log.info("MINOS results:")
    max_label_length = max([len(label) for label in labels])
    minos_unc = minuit_obj.np_merrors()
    for i_par, unc_down, unc_up in zip(range(len(labels)), minos_unc[0], minos_unc[1]):
        # the uncertainties are 0.0 by default if MINOS has not been run
        if unc_up != 0.0 or unc_down != 0.0:
            log.info(
                f"{labels[i_par].ljust(max_label_length)} = "
                f"{minuit_obj.np_values()[i_par]:.4f} -{unc_down:.4f} +{unc_up:.4f}"
            )
