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
        log.info("%s: %f +/- %f", l_with_spacer, bestfit[i], uncertainty[i])


def fit(spec):
    """perform an unconstrained maximum likelihood fit with pyhf and report
    the results of the fit

    Args:
        spec (dict): a pyhf workspace

    Returns:
        (numpy.ndarray, numpy.ndarray, list): best-fit positions of parameters, their uncertainties and names
    """
    log.info("performing unconstrained fit")

    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
    result = pyhf.infer.mle.fit(data, model, return_uncertainties=True)
    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = get_parameter_names(model)

    print_results(bestfit, uncertainty, labels)
    return bestfit, uncertainty, labels
