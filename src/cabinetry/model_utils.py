import logging
from typing import List, Tuple

import pyhf

log = logging.getLogger(__name__)


def get_parameter_names(model: pyhf.pdf.Model) -> List[str]:
    """get the labels of all fit parameters, expanding vectors that act on
    one bin per vector entry (gammas)

    Args:
        model (pyhf.pdf.Model): a HistFactory-style model in ``pyhf`` format

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


def get_asimov_parameters(model: pyhf.pdf.Model) -> Tuple[List[float], List[float]]:
    """Returns a list of Asimov parameter values and pre-fit uncertainties for a model.

    Args:
        model (pyhf.pdf.Model): model for which to extract the parameters

    Returns:
        Tuple[List[float], List[float]]:
            - the Asimov parameters, in the same order as
              ``model.config.suggested_init()``
            - pre-fit uncertainties for the parameters
    """
    # create a list of parameter names, one entry per single parameter
    # (vectors like staterror expanded)
    auxdata_pars_all = []
    for parameter in model.config.auxdata_order:
        auxdata_pars_all += [parameter] * model.config.param_set(parameter).n_parameters

    # create a list of Asimov parameters (constrained parameters at the
    # best-fit value from the aux measurement, unconstrained parameters at
    # the init specified in the workspace)
    asimov_parameters = []
    pre_fit_unc = []  # pre-fit uncertainties for parameters
    for parameter in model.config.par_order:
        # indices in auxdata list that match the current parameter
        aux_indices = [i for i, par in enumerate(auxdata_pars_all) if par == parameter]
        if aux_indices:
            # pick up best-fit value from auxdata
            inits = [
                aux for i, aux in enumerate(model.config.auxdata) if i in aux_indices
            ]
        else:
            # pick up suggested inits (for normfactors)
            inits = model.config.param_set(parameter).suggested_init
        asimov_parameters += inits

        # for constrained parameters, obtain their pre-fit uncertainty
        if model.config.param_set(parameter).constrained:
            pre_fit_unc += model.config.param_set(parameter).width()
        else:
            if model.config.param_set(parameter).n_parameters == 1:
                # unconstrained normfactor, do not add any uncertainties
                pre_fit_unc.append(0.0)
            else:
                # shapefactor
                pre_fit_unc += [0.0] * model.config.param_set(parameter).n_parameters

    return asimov_parameters, pre_fit_unc
