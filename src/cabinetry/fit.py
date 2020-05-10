import logging

import pyhf

from . import histo


log = logging.getLogger(__name__)


def get_parameter_names(model):
    """
    get the labels of all fit parameters, expanding vectors that act on
    one bin per vector entry (gammas)
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
    """
    print the best-fit parameter results and associated uncertainties
    """
    max_label_length = max([len(l) for l in labels])
    for i, l in enumerate(labels):
        l_with_spacer = l + " " * (max_label_length - len(l))
        log.info("%s: %f +/- %f", l_with_spacer, bestfit[i], uncertainty[i])


def fit(spec):
    """
    perform an unconstrained maximum likelihood fit with pyhf and report
    the results of the fit
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
