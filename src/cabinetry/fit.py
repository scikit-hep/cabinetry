import logging

import iminuit
import pyhf

from cabinetry.contrib import matplotlib_visualize

log = logging.getLogger(__name__)


def get_parameter_names(model):
    """get the labels of all fit parameters, expanding vectors that act on
    one bin per vector entry (gammas)

    Args:
        model (pyhf.pdf.Model): a HistFactory-style model in pyhf format

    Returns:
        list: names of fit parameters
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


def print_results(bestfit, uncertainty, labels):
    """print the best-fit parameter results and associated uncertainties

    Args:
        bestfit (numpy.ndarray): best-fit results of parameters
        uncertainty (numpy.ndarray): uncertainties of best-fit parameter results
        labels (list): parameter labels
    """
    max_label_length = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        l_with_spacer = label + " " * (max_label_length - len(label))
        log.info(f"{l_with_spacer}: {bestfit[i]:.6f} +/- {uncertainty[i]:.6f}")


def fit(spec):
    """perform an unconstrained maximum likelihood fit with pyhf and report
    the results of the fit

    Args:
        spec (dict): a pyhf workspace

    Returns:
        tuple: a tuple containing
            - numpy.ndarray: best-fit positions of parameters
            - numpy.ndarray: parameter uncertainties
            - list: parameter names
            - float: -2 log(likelihood) at best-fit point
    """
    log.info("performing unconstrained fit")

    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
    result, best_twice_nll = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_fitted_val=True
    )
    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = get_parameter_names(model)

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {best_twice_nll:.6f} at the best-fit point")
    return bestfit, uncertainty, labels, best_twice_nll


def custom_fit(spec: dict, figure_folder: str) -> None:
    """Perform an unconstrained maximum likelihood fit with iminuit and report
    the result. Compared to fit(), this does not use the pyhf.infer API for more
    control over the minimization.

    Args:
        spec (dict): a pyhf workspace
        figure_folder (str): path to folder where figures will be saved

    Returns:
        tuple: a tuple containing
            - numpy.ndarray: best-fit positions of parameters
            - numpy.ndarray: parameter uncertainties
            - list: parameter names
            - float: -2 log(likelihood) at best-fit point
    """
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))

    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    step_size = [0.1 for _ in init_pars]
    fix_pars = [False for _ in init_pars]

    labels = get_parameter_names(model)

    def twice_nll_func(pars):
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

    corr_mat = m.np_matrix(correlation=True)
    bestfit = m.np_values()
    uncertainty = m.np_errors()
    best_twice_nll = m.fval

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {best_twice_nll:.6f} at the best-fit point")
    matplotlib_visualize.correlation_matrix(corr_mat, labels, figure_folder)
    return bestfit, uncertainty, labels, best_twice_nll
