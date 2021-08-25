import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import iminuit
import numpy as np
import pyhf
import scipy.optimize
import scipy.stats

from cabinetry import model_utils
from cabinetry.fit.results_containers import (
    FitResults,
    LimitResults,
    RankingResults,
    ScanResults,
    SignificanceResults,
)


log = logging.getLogger(__name__)


def print_results(fit_results: FitResults) -> None:
    """Prints the best-fit parameter results and associated uncertainties.

    Args:
        fit_results (FitResults): results of fit to be printed
    """
    max_label_length = max(len(label) for label in fit_results.labels)
    log.info("fit results (with symmetric uncertainties):")
    for i, label in enumerate(fit_results.labels):
        log.info(
            f"{label:<{max_label_length}} = {fit_results.bestfit[i]: .4f} +/- "
            f"{fit_results.uncertainty[i]:.4f}"
        )


def _fit_model_pyhf(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fix_pars: Optional[List[bool]] = None,
    minos: Optional[Union[List[str], Tuple[str, ...]]] = None,
) -> FitResults:
    """Uses the ``pyhf.infer`` API to perform a maximum likelihood fit.

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
        minos (Optional[Union[List[str], Tuple[str, ...]]], optional): runs the MINOS
            algorithm for all parameters specified, defaults to None (does not run
            MINOS)

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer(verbose=1))

    result, corr_mat, best_twice_nll, result_obj = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        fixed_params=fix_pars,
        return_uncertainties=True,
        return_correlations=True,
        return_fitted_val=True,
        return_result_obj=True,
    )
    log.info(f"MINUIT status:\n{result_obj.minuit.fmin}")

    bestfit = pyhf.tensorlib.to_numpy(result[:, 0])
    uncertainty = pyhf.tensorlib.to_numpy(result[:, 1])
    labels = model.config.par_names()
    corr_mat = pyhf.tensorlib.to_numpy(corr_mat)
    best_twice_nll = float(best_twice_nll)  # convert 0-dim np.ndarray to float

    fit_results = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)

    if minos is not None:
        _run_minos(result_obj.minuit, minos, labels)

    return fit_results


def _fit_model_custom(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fix_pars: Optional[List[bool]] = None,
    minos: Optional[Union[List[str], Tuple[str, ...]]] = None,
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
        minos (Optional[Union[List[str], Tuple[str, ...]]], optional): runs the MINOS
            algorithm for all parameters specified, defaults to None (does not run
            MINOS)

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer(verbose=1))

    # use init_pars provided in function argument if they exist, else use default
    init_pars = init_pars or model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    # use fix_pars provided in function argument if they exist, else use default
    fix_pars = fix_pars or model.config.suggested_fixed()

    labels = model.config.par_names()

    # set initial step size to 0 for fixed parameters
    # this will cause the associated parameter uncertainties to be 0 post-fit
    step_size = [0.1 if not fix_pars[i_par] else 0.0 for i_par in range(len(init_pars))]

    def twice_nll_func(pars: np.ndarray) -> Any:
        """The objective for minimization: twice the negative log-likelihood.

        The return value is float-like, but not always a float. The actual type depends
        on the active ``pyhf`` backend.

        Args:
            pars (np.ndarray): parameter values at which the NLL is evaluated

        Returns:
            Any: twice the negative log-likelihood
        """
        twice_nll = -2 * model.logpdf(pars, data)
        return twice_nll[0]

    m = iminuit.Minuit(twice_nll_func, init_pars, name=labels)
    m.errors = step_size
    m.limits = par_bounds
    m.fixed = fix_pars
    m.errordef = 1
    m.print_level = 1

    # decrease tolerance (goal: EDM < 0.002*tol*errordef), default tolerance is 0.1
    m.tol /= 10
    m.migrad()
    m.hesse()
    log.info(f"MINUIT status:\n{m.fmin}")

    bestfit = np.asarray(m.values)
    uncertainty = np.asarray(m.errors)
    corr_mat = m.covariance.correlation()  # iminuit.util.Matrix, subclass of np.ndarray
    best_twice_nll = m.fval

    fit_results = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)

    if minos is not None:
        _run_minos(m, minos, labels)

    return fit_results


def _fit_model(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fix_pars: Optional[List[bool]] = None,
    minos: Optional[Union[List[str], Tuple[str, ...]]] = None,
    custom_fit: bool = False,
) -> FitResults:
    """Interface for maximum likelihood fits through ``pyhf.infer`` API or ``iminuit``.

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
        minos (Optional[Union[List[str], Tuple[str, ...]]], optional): runs the MINOS
            algorithm for all parameters specified, defaults to None (does not run
            MINOS)
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

    Returns:
        FitResults: object storing relevant fit results
    """
    if not custom_fit:
        # use pyhf infer API
        fit_results = _fit_model_pyhf(
            model, data, init_pars=init_pars, fix_pars=fix_pars, minos=minos
        )
    else:
        # use iminuit directly
        fit_results = _fit_model_custom(
            model, data, init_pars=init_pars, fix_pars=fix_pars, minos=minos
        )
    log.debug(f"-2 log(L) = {fit_results.best_twice_nll:.6f} at best-fit point")
    return fit_results


def _run_minos(
    minuit_obj: iminuit.Minuit,
    minos: Union[List[str], Tuple[str, ...]],
    labels: List[str],
) -> None:
    """Determines parameter uncertainties for a list of parameters with MINOS.

    Args:
        minuit_obj (iminuit.Minuit): Minuit instance to use
        minos (Union[List[str], Tuple[str, ...]]): parameters for which MINOS is run
        labels (List[str]]): names of all parameters known to ``iminuit``, these names
            are used in output (may be the same as the names under which ``iminiuit``
            knows parameters)
    """
    for par_name in minos:
        if par_name not in minuit_obj.parameters:
            # parameter not found, skip calculation
            log.warning(f"parameter {par_name} not found in model")
            continue
        log.info(f"running MINOS for {par_name}")
        minuit_obj.minos(par_name)

    log.info("MINOS results:")
    max_label_length = max(len(label) for label in labels)
    minos_unc = [minuit_obj.params[i].merror for i in range(minuit_obj.npar)]
    for i_par, unc in zip(range(len(labels)), minos_unc):
        # if MINOS has not been run, entries are None
        if unc is not None:
            log.info(
                f"{labels[i_par]:<{max_label_length}} = "
                f"{minuit_obj.values[i_par]: .4f} {unc[0]:+.4f} {unc[1]:+.4f}"
            )


def _goodness_of_fit(
    model: pyhf.pdf.Model, data: List[float], best_twice_nll: float
) -> float:
    """Calculates goodness-of-fit p-value with a saturated model.

    Returns NaN if the number of degrees of freedom in the chi2 test is zero (nominal
    fit should already be perfect) or negative (over-parameterized model).

    Args:
        model (pyhf.pdf.Model): model used in the fit for which goodness-of-fit should
            be calculated
        data (List[float]): the observed data
        best_twice_nll (float): best-fit -2 log(likelihood) of fit for which goodness-
            of-fit should be calculated

    Returns:
        float: goodness-of-fit p-value
    """
    if model.config.nauxdata > 0:
        main_data, aux_data = model.fullpdf_tv.split(pyhf.tensorlib.astensor(data))
        # constraint term: log Gaussian(aux_data|parameters) etc.
        constraint_ll = pyhf.tensorlib.to_numpy(
            model.constraint_logpdf(
                aux_data, pyhf.tensorlib.astensor(model.config.suggested_init())
            )
        )
    else:
        # no auxiliary data, so no constraint terms present
        main_data = pyhf.tensorlib.astensor(data)
        constraint_ll = 0.0
    # Poisson term: log Poisson(data|lambda=data), sum is over log likelihood of bins
    poisson_ll = pyhf.tensorlib.to_numpy(
        sum(pyhf.tensorlib.poisson_dist(main_data).log_prob(main_data))
    )
    saturated_nll = -(poisson_ll + constraint_ll)  # saturated likelihood

    log.info("calculating goodness-of-fit")
    delta_nll = best_twice_nll / 2 - saturated_nll
    log.debug(f"Delta NLL = {delta_nll:.6f}")

    # calculate difference in degrees of freedom between fits, given by the number
    # of bins minus the number of unconstrained parameters
    n_dof = sum(
        model.config.channel_nbins.values()
    ) - model_utils.unconstrained_parameter_count(model)
    log.debug(f"number of degrees of freedom: {n_dof}")

    if n_dof <= 0:
        log.warning(
            f"cannot calculate p-value: {n_dof} degrees of freedom and Delta NLL = "
            f"{delta_nll:.6f}"
        )
        return np.nan

    p_val = scipy.stats.chi2.sf(2 * delta_nll, n_dof)
    log.info(f"p-value for goodness-of-fit test: {p_val:.2%}")
    return p_val


def fit(
    model: pyhf.pdf.Model,
    data: List[float],
    minos: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    goodness_of_fit: bool = False,
    custom_fit: bool = False,
) -> FitResults:
    """Performs a  maximum likelihood fit, reports and returns the results.

    Depending on the ``custom_fit`` keyword argument, this uses either the
    ``pyhf.infer`` API or ``iminuit`` directly.

    Args:
        model (pyhf.pdf.Model): model to use in fit
        data (List[float]): data (including auxdata) the model is fit to
        minos (Optional[Union[str, List[str], Tuple[str, ...]]], optional): runs the
            MINOS algorithm for all parameters specified, defaults to None (does not run
            MINOS)
        goodness_of_fit (bool, optional): calculate goodness of fit with a saturated
            model (perfectly fits data with shapefactors in all bins), defaults to False
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

    Returns:
        FitResults: object storing relevant fit results
    """
    log.info("performing maximum likelihood fit")

    # convert minos parameter to list if a single parameter is specified as string
    if isinstance(minos, str):
        minos = [minos]

    # perform fit
    fit_results = _fit_model(model, data, minos=minos, custom_fit=custom_fit)

    print_results(fit_results)

    if goodness_of_fit:
        # calculate goodness-of-fit with saturated model
        p_val = _goodness_of_fit(model, data, fit_results.best_twice_nll)
        fit_results = fit_results._replace(goodness_of_fit=p_val)

    return fit_results


def ranking(
    model: pyhf.pdf.Model,
    data: List[float],
    fit_results: Optional[FitResults] = None,
    custom_fit: bool = False,
) -> RankingResults:
    """Calculates the impact of nuisance parameters on the parameter of interest (POI).

    The impact is given by the difference in the POI between the nominal fit, and a fit
    where the nuisance parameter is held constant at its nominal value plus/minus its
    associated uncertainty. The "pre-fit impact" is obtained by varying the nuisance
    parameters by their uncertainty given by their constraint term.

    Args:
        model (pyhf.pdf.Model): model to use in fits
        data (List[float]): data (including auxdata) the model is fit to
        fit_results (Optional[FitResults], optional): nominal fit results to use for
            ranking, if not specified will repeat nominal fit, defaults to None
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

    Returns:
        RankingResults: fit results for parameters, and pre- and post-fit impacts
    """
    if fit_results is None:
        fit_results = _fit_model(model, data, custom_fit=custom_fit)

    labels = model.config.par_names()
    prefit_unc = model_utils.prefit_uncertainties(model)
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
                fit_results_ranking = _fit_model(
                    model,
                    data,
                    init_pars=init_pars,
                    fix_pars=fix_pars,
                    custom_fit=custom_fit,
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
    model: pyhf.pdf.Model,
    data: List[float],
    par_name: str,
    par_range: Optional[Tuple[float, float]] = None,
    n_steps: int = 11,
    custom_fit: bool = False,
) -> ScanResults:
    """Performs a likelihood scan over the specified parameter.

    If no parameter range is specified, center the scan around the best-fit result for
    the parameter that is being scanned, and scan over twice its uncertainty in each
    direction. The reported likelihood values are the differences between -2 log(L) at
    each point in the scan and the global minimum.

    Args:
        model (pyhf.pdf.Model): model to use in fits
        data (List[float]): data (including auxdata) the model is fit to
        par_name (str): name of parameter to scan over
        par_range (Optional[Tuple[float, float]], optional): upper and lower bounds of
            parameter in scan, defaults to None (automatically determine bounds)
        n_steps (int, optional): number of steps in scan, defaults to 10
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

    Raises:
        ValueError: if parameter is not found in model

    Returns:
        ScanResults: includes parameter name, scanned values and 2*log(likelihood)
        offset
    """
    labels = model.config.par_names()
    init_pars = model.config.suggested_init()
    fix_pars = model.config.suggested_fixed()

    # get index of parameter with name par_name
    par_index = model_utils._parameter_index(par_name, labels)
    if par_index == -1:
        raise ValueError(f"parameter {par_name} not found in model")

    # run a fit with the parameter not held constant, to find the best-fit point
    fit_results = _fit_model(model, data, custom_fit=custom_fit)
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
        scan_fit_results = _fit_model(
            model,
            data,
            init_pars=init_pars_scan,
            fix_pars=fix_pars,
            custom_fit=custom_fit,
        )
        # subtract best-fit
        delta_nlls[i_par] = scan_fit_results.best_twice_nll - nominal_twice_nll

    scan_results = ScanResults(par_name, par_mle, par_unc, scan_values, delta_nlls)
    return scan_results


def limit(
    model: pyhf.pdf.Model,
    data: List[float],
    bracket: Optional[Union[List[float], Tuple[float, float]]] = None,
    tolerance: float = 0.01,
    maxiter: int = 100,
) -> LimitResults:
    """Calculates observed and expected 95% confidence level upper parameter limits.

    Limits are calculated for the parameter of interest (POI) defined in the model.
    Brent's algorithm is used to automatically determine POI values to be tested.

    Args:
        model (pyhf.pdf.Model): model to use in fits
        data (List[float]): data (including auxdata) the model is fit to
        bracket (Optional[Union[List[float], Tuple[float, float]]], optional): the two
            POI values used to start the observed limit determination, the limit must
            lie between these values and the values must not be the same, defaults to
            None (then uses ``0.1`` as default lower value and the upper POI bound
            specified in the measurement as default upper value)
        tolerance (float, optional): tolerance in POI value for convergence to CLs=0.05,
            defaults to 0.01
        maxiter (int, optional): maximum number of steps for limit finding, defaults to
            100

    Raises:
        ValueError: if lower and upper bracket value are the same

    Returns:
        LimitResults: observed and expected limits, CLs values, and scanned points
    """
    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer(verbose=1))

    log.info(f"calculating upper limit for {model.config.poi_name}")

    # set lower POI bound to zero (for use with qmu_tilde)
    par_bounds = model.config.suggested_bounds()
    par_bounds[model.config.poi_index] = [0, par_bounds[model.config.poi_index][1]]
    log.debug("setting lower parameter bound for POI to 0")

    # set default bracket to (0.1, upper POI bound in measurement) if needed
    bracket_left_default = 0.1
    bracket_right_default = par_bounds[model.config.poi_index][1]
    if bracket is None:
        bracket = (bracket_left_default, bracket_right_default)
    elif bracket[0] == bracket[1]:
        raise ValueError(f"the two bracket values must not be the same: {bracket}")

    cache_CLs: Dict[float, tuple] = {}  # cache storing all relevant results

    def _cls_minus_threshold(
        poi: float,
        model: pyhf.pdf.Model,
        data: List[float],
        which_limit: int,
        limit_label: str,
    ) -> float:
        """The root of this function is the POI value at the CLs=0.05 crossing.

        Returns 0.95 for POI values below 0. Makes use of an external cache to avoid
        re-fitting with known POI values and to store all relevant values.

        Args:
            poi (float): value for parameter of interest
            model (pyhf.pdf.Model): model to use in fits
            data (List[float]): data (including auxdata) the model is fit to
            which_limit (int): which limit to run, 0: observed, 1: expected -2 sigma, 2:
                expected -1 sigma, 3: expected, 4: expected +1 sigma, 5: expected +2
                sigma
            limit_label (str): string to use when referring to the current limit

        Returns:
            float: absolute value of difference to CLs=0.05
        """
        if poi <= 0:
            # no fit needed for negative POI value, return a default value
            log.debug(
                f"skipping fit for {model.config.poi_name} = {poi:.4f}, setting CLs = 1"
            )
            return 0.95  # corresponds to distance of CLs = 1 to target CLs = 0.05
        cache = cache_CLs.get(poi)
        if cache:
            observed, expected = cache  # use result from cache
        else:
            # calculate CLs
            results = pyhf.infer.hypotest(
                poi,
                data,
                model,
                test_stat="qtilde",
                return_expected_set=True,
                par_bounds=par_bounds,
            )
            observed = float(results[0])  # 1 value per scan point
            expected = np.asarray(results[1])  # 5 per point (with 1 and 2 sigma bands)
            cache_CLs.update({poi: (observed, expected)})
        current_CLs = np.hstack((observed, expected))[which_limit]
        log.debug(
            f"{model.config.poi_name} = {poi:.4f}, {limit_label} CLs = "
            f"{current_CLs:.4f}{' (cached)' if cache else ''}"
        )
        return current_CLs - 0.05

    # calculate all limits, one by one: observed, expected -2 sigma, expected -1 sigma,
    # expected, expected +1 sigma, expected +2 sigma
    limit_labels = [
        "observed",
        "expected -2 sigma",
        "expected -1 sigma",
        "expected",
        "expected +1 sigma",
        "expected +2 sigma",
    ]
    steps_total = 0
    all_limits = []
    all_converged = True
    for i_limit, limit_label in enumerate(limit_labels):
        log.info(f"determining {limit_label} upper limit")

        try:
            # find the 95% CL upper limit
            res = scipy.optimize.root_scalar(
                _cls_minus_threshold,
                bracket=bracket,
                args=(model, data, i_limit, limit_label),
                method="brentq",
                options={"xtol": tolerance, "maxiter": maxiter},
            )
        except ValueError:
            # invalid starting bracket is most common issue
            log.error(
                f"CLs values at {bracket[0]:.4f} and {bracket[1]:.4f} do not bracket "
                "CLs=0.05, try a different starting bracket"
            )
            raise

        if not res.converged:
            log.error(
                f"failed to converge after {res.function_calls} steps: {res.flag}"
            )
            all_converged = False
        else:
            log.info(f"successfully converged after {res.function_calls} steps")

        log.info(f"{limit_label} upper limit: {res.root:.4f}")
        all_limits.append(res.root)
        steps_total += res.function_calls

        # determine the starting bracket for the next limit calculation
        if i_limit < 5:
            # expected CLs values for next limit type that have been calculated already
            exp_CLs_next = np.asarray([exp[i_limit] for _, exp in cache_CLs.values()])
            # associated POI values
            poi_arr = np.fromiter(cache_CLs.keys(), dtype=float)

            # left: CLs has to be > 0.05, mask out values where CLs <= 0.05
            masked_CLs_left = np.where(exp_CLs_next <= 0.05, 1, exp_CLs_next)
            if sum(masked_CLs_left != 1) == 0:
                # all values are below 0.05, pick default lower bound
                bracket_left = bracket_left_default
            else:
                # find closest to CLs = 0.05 from above
                bracket_left = poi_arr[np.argmin(masked_CLs_left)]

            # right: CLs has to be < 0.05, mask out values where CLs >= 0.05
            masked_CLs_right = np.where(exp_CLs_next >= 0.05, -1, exp_CLs_next)
            if sum(masked_CLs_right != -1) == 0:
                # all values are above 0.05, pick default upper bound
                bracket_right = bracket_right_default
            else:
                # find closest to CLs=0.05 from below
                bracket_right = poi_arr[np.argmax(masked_CLs_right)]

            bracket = (bracket_left, bracket_right)

    # report all results
    log.info(f"total of {steps_total} steps to calculate all limits")
    if not all_converged:
        log.error("one or more calculations did not converge, check log")
    log.info("summary of upper limits:")
    for i_limit, limit_label in enumerate(limit_labels):
        log.info(f"{limit_label.ljust(17)}: {all_limits[i_limit]:.4f}")

    # sort all CLs values and scanned POI points by increasing POI value
    poi_arr = np.fromiter(cache_CLs.keys(), dtype=float)
    sorted_indices = np.argsort(poi_arr)
    observed_CLs_np = np.asarray([obs for obs, _ in cache_CLs.values()])[sorted_indices]
    expected_CLs_np = np.asarray([exp for _, exp in cache_CLs.values()])[sorted_indices]
    poi_arr = poi_arr[sorted_indices]

    limit_results = LimitResults(
        all_limits[0],
        np.asarray(all_limits[1:]),
        observed_CLs_np,
        expected_CLs_np,
        poi_arr,
    )
    return limit_results


def significance(model: pyhf.pdf.Model, data: List[float]) -> SignificanceResults:
    """Calculates the discovery significance of a positive signal.

    Observed and expected p-values and significances are both calculated and reported.

    Args:
        model (pyhf.pdf.Model): model to use in fits
        data (List[float]): data (including auxdata) the model is fit to

    Returns:
        SignificanceResults: observed and expected p-values and significances
    """
    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer(verbose=1))

    log.info("calculating discovery significance")
    obs_p_val, exp_p_val = pyhf.infer.hypotest(
        0.0, data, model, test_stat="q0", return_expected=True
    )
    obs_p_val = float(obs_p_val)
    exp_p_val = float(exp_p_val)
    obs_significance = scipy.stats.norm.isf(obs_p_val, 0, 1)
    exp_significance = scipy.stats.norm.isf(exp_p_val, 0, 1)

    if obs_p_val >= 1e-3:
        log.info(f"observed p-value: {obs_p_val:.3%}")
    else:
        log.info(f"observed p-value: {obs_p_val:.3e}")
    log.info(f"observed significance: {obs_significance:.3f}")
    if exp_p_val >= 1e-3:
        log.info(f"expected p-value: {exp_p_val:.3%}")
    else:
        log.info(f"expected p-value: {exp_p_val:.3e}")
    log.info(f"expected significance: {exp_significance:.3f}")

    significance_results = SignificanceResults(
        obs_p_val, obs_significance, exp_p_val, exp_significance
    )
    return significance_results
