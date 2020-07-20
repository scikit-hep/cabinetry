import logging

import pyhf


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
    log.info(f"performing unconstrained fit")

    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
    result, twice_nll = pyhf.infer.mle.fit(
        data, model, return_uncertainties=True, return_fitted_val=True
    )
    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = get_parameter_names(model)

    print_results(bestfit, uncertainty, labels)
    log.debug(f"-2 log(L) = {twice_nll:.6f} at the best-fit point")
    return bestfit, uncertainty, labels, twice_nll
