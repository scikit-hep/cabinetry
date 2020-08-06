import logging
from typing import Any, Dict, List, Tuple

import iminuit
import numpy as np
import pyhf


log = logging.getLogger(__name__)


def get_parameter_names(model: pyhf.pdf.Model) -> List[str]:
    """get the labels of all fit parameters, expanding vectors that act on
    one bin per vector entry (gammas)

    Args:
        model (pyhf.pdf.Model): a HistFactory-style model in pyhf format

    Returns:
        List[str]: names of fit parameters
    """
    labels = []
    for parname in model.config.par_order:
        for i_par in range(model.config.param_set(parname).n_parameters):
            labels.append(
                "{}[bin_{}]".format(parname, i_par)
                if model.config.param_set(parname).n_parameters > 1
                else parname
            )
    return labels


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
        log.info(f"{l_with_spacer}: {bestfit[i]:.6f} +/- {uncertainty[i]:.6f}")


def fit(
    spec: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, List[str], float, np.ndarray]:
    """perform an unconstrained maximum likelihood fit with pyhf and report
    the results of the fit

    Args:
        spec (Dict[str, Any]): a pyhf workspace specificaton

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], float, np.ndarray]:
            - best-fit positions of parameters
            - parameter uncertainties
            - parameter names
            - -2 log(likelihood) at best-fit point
            - correlation matrix
    """
    log.info("performing unconstrained fit")

    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
    result, result_obj = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_result_obj=True
    )

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    best_twice_nll = float(result_obj.fun)  # convert 0-dim np.ndarray to float
    corr_mat = result_obj.minuit.np_matrix(correlation=True)
    labels = get_parameter_names(model)

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {best_twice_nll:.6f} at the best-fit point")
    return bestfit, uncertainty, labels, best_twice_nll, corr_mat


def custom_fit(
    spec: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, List[str], float, np.ndarray]:
    """Perform an unconstrained maximum likelihood fit with iminuit and report
    the result. Compared to fit(), this does not use the pyhf.infer API for more
    control over the minimization.

    Args:
        spec (Dict[str, Any]): a pyhf workspace specificaton

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], float, np.ndarray]:
            - best-fit positions of parameters
            - parameter uncertainties
            - parameter names
            - -2 log(likelihood) at best-fit point
            - correlation matrix
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fix_pars = model.config.suggested_fixed()

    step_size = [0.1 for _ in init_pars]

    labels = get_parameter_names(model)

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

    corr_mat = m.np_matrix(correlation=True)
    bestfit = m.np_values()
    uncertainty = m.np_errors()
    best_twice_nll = m.fval

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {best_twice_nll:.6f} at the best-fit point")

    return bestfit, uncertainty, labels, best_twice_nll, corr_mat
