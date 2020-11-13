import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import iminuit
import numpy as np
import pyhf
import scipy.optimize
import scipy.stats

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
        goodess_of_fit (float, optional): goodness-of-fit p-value, defaults to -1
    """

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List[str]
    corr_mat: np.ndarray
    best_twice_nll: float
    goodness_of_fit: float = -1


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
        parameter_values (np.ndarray): parameter values used in scan
        delta_nlls (np.ndarray): -2 log(L) difference at each scanned point
    """

    name: str
    bestfit: float
    uncertainty: float
    parameter_values: np.ndarray
    delta_nlls: np.ndarray


class LimitResults(NamedTuple):
    """Collects parameter upper limit results in one object.

    Args:
        observed_limit (np.ndarray): observed limit
        expected_limit (np.ndarray): expected limit, including 1 and 2 sigma bands
        observed_CLs (np.ndarray): observed CLs values
        expected_CLs (np.ndarray): expected CLs values, including 1 and 2 sigma bands
        poi_values (np.ndarray): POI values used in scan
    """

    observed_limit: float
    expected_limit: np.ndarray
    observed_CLs: np.ndarray
    expected_CLs: np.ndarray
    poi_values: np.ndarray


def print_results(
    fit_results: FitResults,
) -> None:
    """Prints the best-fit parameter results and associated uncertainties.

    Args:
        fit_results (FitResults): results of fit to be printed
    """
    max_label_length = max([len(label) for label in fit_results.labels])
    log.info("fit results (with symmetric uncertainties):")
    for i, label in enumerate(fit_results.labels):
        log.info(
            f"{label.ljust(max_label_length)}: {fit_results.bestfit[i]: .4f} +/- "
            f"{fit_results.uncertainty[i]:.4f}"
        )


def _fit_model_pyhf(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fix_pars: Optional[List[bool]] = None,
    minos: Optional[List[str]] = None,
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
        minos (Optional[List[str]], optional): runs the MINOS algorithm for all
            parameters specified in the list, defaults to None (does not run MINOS)

    Returns:
        FitResults: object storing relevant fit results
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    result, result_obj = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        fixed_params=fix_pars,
        return_uncertainties=True,
        return_result_obj=True,
    )

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = model_utils.get_parameter_names(model)
    corr_mat = result_obj.minuit.np_matrix(correlation=True, skip_fixed=False)
    best_twice_nll = float(result_obj.fun)  # convert 0-dim np.ndarray to float

    fit_results = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)

    if minos is not None:
        parameters_translated = []
        for minos_par in minos:
            par_index = model_utils._get_parameter_index(minos_par, labels)
            if par_index != -1:
                # pyhf does not hand over parameter names, all parameters are known as
                # x0, x1, etc.
                parameters_translated.append(f"x{par_index}")

        _run_minos(result_obj.minuit, parameters_translated, labels)

    return fit_results


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

    def twice_nll_func(pars: np.ndarray) -> float:
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

    fit_results = FitResults(bestfit, uncertainty, labels, corr_mat, best_twice_nll)

    if minos is not None:
        _run_minos(m, minos, labels)

    return fit_results


def _fit_model(
    model: pyhf.pdf.Model,
    data: List[float],
    init_pars: Optional[List[float]] = None,
    fix_pars: Optional[List[bool]] = None,
    minos: Optional[List[str]] = None,
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
        minos (Optional[List[str]], optional): runs the MINOS algorithm for all
            parameters specified in the list, defaults to None (does not run MINOS)
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


def _run_minos(minuit_obj: iminuit.Minuit, minos: List[str], labels: List[str]) -> None:
    """Determines parameter uncertainties for a list of parameters with MINOS.

    Args:
        minuit_obj (iminuit.Minuit): Minuit instance to use
        minos (List[str]): parameters for which MINOS is run
        labels (List[str]]): names of all parameters known to ``iminuit``, these names
            are used in output (may be the same as the names under which ``iminiuit``
            knows parameters)
    """
    for par_name in minos:
        # get index of current parameter in labels (to translate its name if iminuit
        # did not receive the parameter labels)
        par_index = model_utils._get_parameter_index(par_name, minuit_obj.parameters)
        if par_index == -1:
            # parameter not found, skip calculation
            continue
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


def _goodness_of_fit(
    model: pyhf.pdf.Model, data: List[float], best_twice_nll: float
) -> float:
    """Calculates goodness-of-fit p-value with a saturated model.

    Args:
        model (pyhf.pdf.Model): model used in the fit for which goodness-of-fit should
            be calculated
        data (List[float]): the observed data
        best_twice_nll (float): best-fit -2 log(likelihood) of fit for which goodness-
            of-fit should be calculated

    Returns:
        float: goodness-of-fit p-value
    """
    main_data, aux_data = model.fullpdf_tv.split(pyhf.tensorlib.astensor(data))
    # Poisson term: log Poisson(data|lambda=data), sum is over log likelihood of bins
    poisson_ll = sum(pyhf.tensorlib.poisson_dist(main_data).log_prob(main_data))
    # constraint term: log Gaussian(aux_data|parameters) etc.
    constraint_ll = model.constraint_logpdf(
        aux_data, pyhf.tensorlib.astensor(model.config.suggested_init())
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
    p_val = scipy.stats.chi2.sf(2 * delta_nll, n_dof)
    log.info(f"p-value for goodness-of-fit test: {p_val*100:.2f}%")
    return p_val


def fit(
    spec: Dict[str, Any],
    asimov: bool = False,
    minos: Optional[Union[str, List[str]]] = None,
    goodness_of_fit: bool = False,
    custom_fit: bool = False,
) -> FitResults:
    """Performs a  maximum likelihood fit, reports and returns the results.

    The ``asimov`` flag allows to fit the Asimov dataset instead of observed data.
    Depending on the ``custom_fit`` keyword argument, this uses either the
    ``pyhf.infer`` API or ``iminuit`` directly for more control over the minimization.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to fit the Asimov dataset, defaults to False
        minos (Optional[Union[str, List[str]]], optional): runs the MINOS algorithm for
            all parameters specified in the list, defaults to None (does not run MINOS)
        goodness_of_fit (bool, optional): calculate goodness of fit with a saturated
            model (perfectly fits data with shapefactors in all bins), defaults to False
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

    Returns:
        FitResults: object storing relevant fit results
    """
    log.info("performing maximum likelihood fit")

    model, data = model_utils.model_and_data(spec, asimov=asimov)

    # convert minos parameter to list, if a parameter is specified and is not a list
    if minos is not None and not isinstance(minos, list):
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
    spec: Dict[str, Any],
    fit_results: FitResults,
    asimov: bool = False,
    custom_fit: bool = False,
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
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

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
    spec: Dict[str, Any],
    par_name: str,
    par_range: Optional[Tuple[float, float]] = None,
    n_steps: int = 11,
    asimov: bool = False,
    custom_fit: bool = False,
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
        custom_fit (bool, optional): whether to use the ``pyhf.infer`` API or
            ``iminuit``, defaults to False (using ``pyhf.infer``)

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
    par_index = model_utils._get_parameter_index(par_name, labels)
    if par_index == -1:
        raise ValueError(f"could not find parameter {par_name} in model")

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
    spec: Dict[str, Any],
    bracket: Optional[List[float]] = None,
    asimov: bool = False,
    tolerance: float = 0.01,
) -> LimitResults:
    """Calculates observed and expected 95% confidence level upper parameter limits.

    Limits are calculated for the parameter of interest (POI) defined in the workspace.
    Brent's algorithm is used to automatically determine POI values to be tested.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        bracket (Optional[List[float]], optional): the two POI values used to start the
            observed limit determination, example: ``[1.0, 2.0]`` (final POI values may
            lie outside this bracket), the two values must not be the same, defaults to
            None (then uses ``[0.5, 1.5]`` as default)
        asimov (bool, optional): whether to fit the Asimov dataset, defaults to False
        tolerance (float, optional): tolerance for convergence to CLs=0.05, defaults to
            0.01

    Raises:
        ValueError: if lower and upper bracket value are the same

    Returns:
        LimitResults: observed and expected limits, CLs values, and scanned points
    """
    if bracket is None:
        bracket = [0.5, 1.5]
    elif bracket[0] == bracket[1]:
        raise ValueError(f"the two bracket values must not be the same: " f"{bracket}")

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=False))
    model, data = model_utils.model_and_data(spec, asimov=asimov)

    log.info(f"calculating upper limit for {model.config.poi_name}")

    # set lower POI bound to zero (for use with qmu_tilde)
    par_bounds = model.config.suggested_bounds()
    par_bounds[model.config.poi_index] = [0, par_bounds[model.config.poi_index][1]]
    log.debug("setting lower parameter bound for POI to 0")

    poi_list = []  # scanned POI values
    observed_CLs_list = []  # observed CLs values, one entry per scan point
    expected_CLs_list = []  # expected CLs values, 5 per point (with 1 and 2 sigma band)

    def _CLs_distance_to_crossing(
        poi: float,
        data: List[float],
        model: pyhf.pdf.Model,
        which_limit: int,
        limit_label: str,
    ) -> float:
        """Objective function to minimize in order to find CLs=0.05 crossing.

        Each observed and expected CLs result is also appended to lists, useful for
        visualization and to optimize the starting bracket for subsequent calculations.
        For POI values below 0, returns the maximum possible distance of 0.95.

        Args:
            poi (float): value for parameter of interest
            data (List[float]): data to fit to
            model (pyhf.pdf.Model): model to fit to data
            which_limit (int): which limit to run, 0: observed, 1: expected -2 sigma, 2:
                expected -1 sigma, 3: expected, 4: expected +1 sigma, 5: expected +2
                sigma
            limit_label (str): string to use when referring to the current limit

        Returns:
            float: absolute value of difference to CLs=0.05
        """
        if poi < 0:
            # no fit needed for negative POI value, return a default value
            log.debug(
                f"optimizer used {model.config.poi_name} = {poi:.4f}, skipping fit and "
                f"setting CLs = 1"
            )
            return 0.95  # corresponds to distance of CLs = 1 to target CLs = 0.05
        results = pyhf.infer.hypotest(
            poi,
            data,
            model,
            qtilde=True,
            return_expected_set=True,
            par_bounds=par_bounds,
        )
        observed = float(results[0])
        expected = np.asarray(results[1])
        poi_list.append(poi)
        observed_CLs_list.append(observed)
        expected_CLs_list.append(expected)
        current_CLs = np.hstack((observed, expected))[which_limit]
        log.debug(
            f"{model.config.poi_name} = {poi:.4f}, {limit_label} CLs = "
            f"{current_CLs:.4f}"
        )
        return np.abs(current_CLs - 0.05)

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

        # find the 95% CL upper limit
        res = scipy.optimize.minimize_scalar(
            _CLs_distance_to_crossing,
            bracket=bracket,
            args=(data, model, i_limit, limit_label),
            method="brent",
            options={"xtol": tolerance, "maxiter": 100},
        )
        if (not res.success) or (res.fun > tolerance):
            log.error(
                f"failed to converge after {res.nfev} steps, distance from CLS=0.05 is "
                f"{res.fun:.4f}"
            )
            all_converged = False
        else:
            log.info(f"successfully converged after {res.nfev} steps")

        log.info(f"{limit_label} upper limit: {res.x:.4f}")
        all_limits.append(res.x)
        steps_total += res.nfev

        # determine the starting bracket for the next limit calculation
        if i_limit < 5:
            # get sorted list of POI values and associated expected CLs
            sorted_indices = np.argsort(poi_list)
            expected_CLs_np = np.asarray(expected_CLs_list)[sorted_indices]
            poi_list_np = np.asarray(poi_list)[sorted_indices]

            # interpolate to get expected CLs=0.06 and CLs=0.04 positions, inverted as
            # np.interp expects function to increase
            # for i_limit = 0, the next limit will be expected -2 sigma, corresponding
            # to expected_CLs_np[:, 0] etc.
            next_bracket: List[float] = np.interp(
                [0.06, 0.04], expected_CLs_np[:, i_limit][::-1], poi_list_np[::-1]
            ).tolist()
            # if the interpolation fails and the lower/upper bound are the same, then
            # offset both values to avoid getting stuck
            if next_bracket[0] == next_bracket[1]:
                next_bracket = [next_bracket[0] - 1, next_bracket[1] + 1]
            bracket = next_bracket

    # report all results
    log.info(f"total of {steps_total} steps to calculate all limits")
    if not all_converged:
        log.error("one or more calculations did not converge, check log")
    log.info("summary of upper limits:")
    for i_limit, limit_label in enumerate(limit_labels):
        log.info(f"{limit_label.ljust(17)}: {all_limits[i_limit]:.4f}")

    # sort all CLs values and scanned POI points by increasing POI value
    sorted_indices = np.argsort(poi_list)
    observed_CLs_np = np.asarray(observed_CLs_list)[sorted_indices]
    expected_CLs_np = np.asarray(expected_CLs_list)[sorted_indices]
    poi_list_np = np.asarray(poi_list)[sorted_indices]

    limit_results = LimitResults(
        all_limits[0],
        np.asarray(all_limits[1:]),
        observed_CLs_np,
        expected_CLs_np,
        poi_list_np,
    )
    return limit_results
