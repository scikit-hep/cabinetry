import logging
from typing import Any, Dict, List, Tuple

import awkward1 as ak
import numpy as np
import pyhf


log = logging.getLogger(__name__)


def model_and_data(
    spec: Dict[str, Any], asimov: bool = False, with_aux: bool = True
) -> Tuple[pyhf.pdf.Model, List[float]]:
    """Returns model and data for a ``pyhf`` workspace specification.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to return the Asimov dataset, defaults
            to False
        with_aux (bool, optional): whether to also return auxdata, defaults
            to True

    Returns:
        Tuple[pyhf.pdf.Model, List[float]]:
            - a HistFactory-style model in ``pyhf`` format
            - the data (plus auxdata if requested) for the model
    """
    workspace = pyhf.Workspace(spec)
    model = workspace.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        }
    )  # use HistFactory InterpCode=4
    if not asimov:
        data = workspace.data(model, with_aux=with_aux)
    else:
        data = build_Asimov_data(model, with_aux=with_aux)
    return model, data


def get_parameter_names(model: pyhf.pdf.Model) -> List[str]:
    """Returns the labels of all fit parameters.

    Vectors that act on one bin per vector entry (gammas) are expanded.

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


def build_Asimov_data(model: pyhf.Model, with_aux: bool = True) -> List[float]:
    """Returns the Asimov dataset (optionally with auxdata) for a model.

    Args:
        model (pyhf.Model): the model from which to construct the
            dataset
        with_aux (bool, optional): whether to also return auxdata, defaults
            to True

    Returns:
        List[float]: the Asimov dataset
    """
    asimov_data = np.sum(model.nominal_rates, axis=1)[0][0].tolist()
    if with_aux:
        return asimov_data + model.config.auxdata
    return asimov_data


def get_asimov_parameters(model: pyhf.pdf.Model) -> np.ndarray:
    """Returns a list of Asimov parameter values for a model.

    Args:
        model (pyhf.pdf.Model): model for which to extract the parameters

    Returns:
        np.ndarray: the Asimov parameters, in the same order as
            ``model.config.suggested_init()``
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

    return np.asarray(asimov_parameters)


def get_prefit_uncertainties(model: pyhf.pdf.Model) -> np.ndarray:
    """Returns a list of pre-fit parameter uncertainties for a model.

    For unconstrained parameters the uncertainty is set to 0.

    Args:
        model (pyhf.pdf.Model): model for which to extract the parameters

    Returns:
        np.ndarray: pre-fit uncertainties for the parameters, in the same
            order as ``model.config.suggested_init()``
    """
    pre_fit_unc = []  # pre-fit uncertainties for parameters
    for parameter in model.config.par_order:
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
    return np.asarray(pre_fit_unc)


def calculate_stdev(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
) -> ak.highlevel.Array:
    """Calculates the symmetrized yield standard deviation of a model.

    Args:
        model (pyhf.pdf.Model): the model for which to calculate the standard
            deviations for all bins
        parameters (np.ndarray): central values of model parameters
        uncertainty (np.ndarray): uncertainty of model parameters
        corr_mat (np.ndarray): correlation matrix

    Returns:
        ak.highlevel.Array: array of channels, each channel
        is an array of standard deviations per bin
    """
    # indices where to split to separate all bins into regions
    # last index dropped since no extra split is needed after the last bin
    region_split = [model.config.channel_nbins[ch] for ch in model.config.channels][:-1]

    # the lists up_variations and down_variations will contain the model distributions
    # with all parameters varied individually within uncertainties
    # indices: variation, channel, bin
    up_variations = []
    down_variations = []

    # calculate the model distribution for every parameter varied up and down
    # within the respective uncertainties
    for i_par in range(model.config.npars):
        # central parameter values, but one parameter varied within uncertainties
        up_pars = parameters.copy()
        up_pars[i_par] += uncertainty[i_par]
        down_pars = parameters.copy()
        down_pars[i_par] -= uncertainty[i_par]

        # total model distribution with this parameter varied up
        up_combined = model.expected_data(up_pars, include_auxdata=False)
        up_yields = np.split(up_combined, region_split)
        up_variations.append(up_yields)

        # total model distribution with this parameter varied down
        down_combined = model.expected_data(down_pars, include_auxdata=False)
        down_yields = np.split(down_combined, region_split)
        down_variations.append(down_yields)

    # convert to awkward arrays for further processing
    up_variations = ak.from_iter(up_variations)
    down_variations = ak.from_iter(down_variations)

    # total variance, indices are: channel, bin
    total_variance_list = [
        np.zeros(shape=(model.config.channel_nbins[ch])) for ch in model.config.channels
    ]  # list of arrays, each array has as many entries as there are bins
    total_variance = ak.from_iter(total_variance_list)

    # loop over parameters to sum up total variance
    # first do the diagonal of the correlation matrix
    for i_par in range(model.config.npars):
        symmetric_uncertainty = (up_variations[i_par] - down_variations[i_par]) / 2
        total_variance = total_variance + symmetric_uncertainty ** 2

    labels = get_parameter_names(model)
    # continue with off-diagonal contributions if there are any
    if np.count_nonzero(corr_mat - np.diag(np.ones_like(parameters))) > 0:
        # loop over pairs of parameters
        for i_par in range(model.config.npars):
            for j_par in range(model.config.npars):
                if j_par >= i_par:
                    continue  # only loop over the half the matrix due to symmetry
                corr = corr_mat[i_par, j_par]
                # an approximate calculation could be done here by requiring
                # e.g. abs(corr) > 1e-5 to continue
                if (
                    labels[i_par][0:10] == "staterror_"
                    and labels[j_par][0:10] == "staterror_"
                ):
                    continue  # two different staterrors are orthogonal and will not contribute
                sym_unc_i = (up_variations[i_par] - down_variations[i_par]) / 2
                sym_unc_j = (up_variations[j_par] - down_variations[j_par]) / 2
                # factor of two below is there since loop is only over half the matrix
                total_variance = total_variance + 2 * (corr * sym_unc_i * sym_unc_j)

    # convert to standard deviation
    total_stdev = np.sqrt(total_variance)
    log.debug(f"total stdev is {total_stdev}")
    return total_stdev
