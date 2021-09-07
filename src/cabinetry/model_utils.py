import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import awkward as ak
import numpy as np
import pyhf

from cabinetry.fit.results_containers import FitResults


log = logging.getLogger(__name__)


# cache holding results from yield uncertainty calculations
_YIELD_STDEV_CACHE: Dict[Any, Tuple[List[List[float]], List[float]]] = {}


class ModelPrediction(NamedTuple):
    """Model prediction with yields and total uncertainties per bin and channel.

    Args:
        model (pyhf.pdf.Model): model to which prediction corresponds to
        model_yields (List[List[List[float]]]): yields per sample, channel and bin,
            indices: channel, sample, bin
        total_stdev_model_bins (List[List[float]]): total yield uncertainty per channel
            and per bin, indices: channel, bin
        total_stdev_model_channels (List[float]): total yield uncertainty per channel,
            index: channel
        label (str): label for the prediction, e.g. "pre-fit" or "post-fit"
    """

    model: pyhf.pdf.Model
    model_yields: List[List[List[float]]]
    total_stdev_model_bins: List[List[float]]
    total_stdev_model_channels: List[float]
    label: str


def model_and_data(
    spec: Dict[str, Any], asimov: bool = False, include_auxdata: bool = True
) -> Tuple[pyhf.pdf.Model, List[float]]:
    """Returns model and data for a ``pyhf`` workspace specification.

    Args:
        spec (Dict[str, Any]): a ``pyhf`` workspace specification
        asimov (bool, optional): whether to return the Asimov dataset, defaults to False
        include_auxdata (bool, optional): whether to also return auxdata, defaults to
            True

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
    )  # use HistFactory InterpCode=4 (default in pyhf since v0.6.0)
    if not asimov:
        data = workspace.data(model, include_auxdata=include_auxdata)
    else:
        data = asimov_data(model, include_auxdata=include_auxdata)
    return model, data


def asimov_data(model: pyhf.Model, include_auxdata: bool = True) -> List[float]:
    """Returns the Asimov dataset (optionally with auxdata) for a model.

    Initial parameter settings for normalization factors in the workspace are treated as
    the default settings for that parameter. Fitting the Asimov dataset will recover
    these initial settings as the maximum likelihood estimate for normalization factors.
    Initial settings for other modifiers are ignored.

    Args:
        model (pyhf.Model): the model from which to construct the dataset
        include_auxdata (bool, optional): whether to also return auxdata, defaults to
            True

    Returns:
        List[float]: the Asimov dataset
    """
    asimov_data = pyhf.tensorlib.tolist(
        model.expected_data(asimov_parameters(model), include_auxdata=include_auxdata)
    )
    return asimov_data


def asimov_parameters(model: pyhf.pdf.Model) -> np.ndarray:
    """Returns a list of Asimov parameter values for a model.

    For normfactors and shapefactors, initial parameter settings (specified in the
    workspace) are treated as nominal settings. This ignores custom auxiliary data set
    in the measurement configuration in the workspace.

    Args:
        model (pyhf.pdf.Model): model for which to extract the parameters

    Returns:
        np.ndarray: the Asimov parameters, in the same order as
        ``model.config.suggested_init()``
    """
    # create a list of Asimov parameters (constrained parameters at best-fit value from
    # the aux measurement, unconstrained parameters at init specified in the workspace)
    asimov_parameters = []
    for parameter in model.config.par_order:
        if not model.config.param_set(parameter).constrained:
            # unconstrained parameter: use suggested inits (for normfactor/shapefactor)
            inits = model.config.param_set(parameter).suggested_init
        elif dict(model.config.modifiers)[parameter] in ["histosys", "normsys"]:
            # histosys/normsys: Gaussian constraint, nominal value 0
            inits = [0.0] * model.config.param_set(parameter).n_parameters
        else:
            # remaining modifiers are staterror/lumi with Gaussian constraint, and
            # shapesys with Poisson constraint, all have nominal value of 1
            inits = [1.0] * model.config.param_set(parameter).n_parameters

        asimov_parameters += inits

    return np.asarray(asimov_parameters)


def prefit_uncertainties(model: pyhf.pdf.Model) -> np.ndarray:
    """Returns a list of pre-fit parameter uncertainties for a model.

    For unconstrained parameters the uncertainty is set to 0. It is also set to 0 for
    fixed parameters (similarly to how the post-fit uncertainties are defined to be 0).

    Args:
        model (pyhf.pdf.Model): model for which to extract the parameters

    Returns:
        np.ndarray: pre-fit uncertainties for the parameters, in the same order as
        ``model.config.suggested_init()``
    """
    pre_fit_unc = []  # pre-fit uncertainties for parameters
    for parameter in model.config.par_order:
        # obtain pre-fit uncertainty for constrained, non-fixed parameters
        if (
            model.config.param_set(parameter).constrained
            and not model.config.param_set(parameter).suggested_fixed
        ):
            pre_fit_unc += model.config.param_set(parameter).width()
        else:
            if model.config.param_set(parameter).n_parameters == 1:
                # unconstrained normfactor or fixed parameter, uncertainty is 0
                pre_fit_unc.append(0.0)
            else:
                # shapefactor
                pre_fit_unc += [0.0] * model.config.param_set(parameter).n_parameters
    return np.asarray(pre_fit_unc)


def _channel_boundary_indices(model: pyhf.pdf.Model) -> List[int]:
    """Returns indices for splitting a concatenated list of observations into channels.

    This is useful in combination with ``pyhf.pdf.Model.expected_data``, which returns
    the yields across all bins in all channels. These indices mark the positions where a
    channel begins. No index is returned for the first channel, which begins at ``[0]``.
    The returned indices can be used with ``numpy.split``.

    Args:
        model (pyhf.pdf.Model): the model that defines the channels

    Returns:
        List[int]: indices of positions where a channel begins, no index is included for
        the first bin of the first channel (which is always at ``[0]``)
    """
    # get the amount of bins per channel
    bins_per_channel = [model.config.channel_nbins[ch] for ch in model.config.channels]
    # indices of positions where a new channel starts (from the second channel onwards)
    channel_start = [sum(bins_per_channel[:i]) for i in range(1, len(bins_per_channel))]
    return channel_start


def yield_stdev(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
) -> Tuple[List[List[float]], List[float]]:
    """Calculates symmetrized yield standard deviation of a model, per bin and channel.

    Returns both the uncertainties per bin (in a list of channels), and the uncertainty
    of the total yield per channel (again, for a list of channels). To calculate the
    uncertainties for the total yield, the function internally treats the sum of yields
    per channel like another channel with one bin. The results of this function are
    cached to speed up subsequent calls with the same arguments.

    Args:
        model (pyhf.pdf.Model): the model for which to calculate the standard deviations
            for all bins
        parameters (np.ndarray): central values of model parameters
        uncertainty (np.ndarray): uncertainty of model parameters
        corr_mat (np.ndarray): correlation matrix

    Returns:
        Tuple[List[List[float]], List[float]]:
            - list of channels, each channel is a list of standard deviations per bin
            - list of standard deviations per channel
    """
    # check whether results are already stored in cache
    cached_results = _YIELD_STDEV_CACHE.get(
        (model, tuple(parameters), tuple(uncertainty), corr_mat.data.tobytes()), None
    )
    if cached_results is not None:
        # return results from cache
        return cached_results

    # indices where to split to separate all bins into regions
    region_split_indices = _channel_boundary_indices(model)

    # the lists up_variations and down_variations will contain the model distributions
    # with all parameters varied individually within uncertainties
    # indices: variation, channel, bin
    # following the channels contained in the model, there are additional entries with
    # yields summed per channel (internally treated like additional channels) to get the
    # per-channel uncertainties
    up_variations = []
    down_variations = []

    # calculate the model distribution for every parameter varied up and down
    # within the respective uncertainties
    for i_par in range(model.config.npars):
        # central parameter values, but one parameter varied within uncertainties
        up_pars = parameters.copy().astype(float)  # ensure float for correct addition
        up_pars[i_par] += uncertainty[i_par]
        down_pars = parameters.copy().astype(float)
        down_pars[i_par] -= uncertainty[i_par]

        # total model distribution with this parameter varied up
        up_combined = pyhf.tensorlib.to_numpy(
            model.expected_data(up_pars, include_auxdata=False)
        )
        up_yields = np.split(up_combined, region_split_indices)
        # append list of yields summed per channel
        up_yields += [np.asarray([sum(chan_yields)]) for chan_yields in up_yields]
        up_variations.append(up_yields)

        # total model distribution with this parameter varied down
        down_combined = pyhf.tensorlib.to_numpy(
            model.expected_data(down_pars, include_auxdata=False)
        )
        down_yields = np.split(down_combined, region_split_indices)
        # append list of yields summed per channel
        down_yields += [np.asarray([sum(chan_yields)]) for chan_yields in down_yields]
        down_variations.append(down_yields)

    # convert to awkward arrays for further processing
    up_variations = ak.from_iter(up_variations)
    down_variations = ak.from_iter(down_variations)

    # total variance, indices are: channel, bin
    n_channels = len(model.config.channels)
    total_variance_list = [
        np.zeros(model.config.channel_nbins[ch]) for ch in model.config.channels
    ]  # list of arrays, each array has as many entries as there are bins
    # append placeholders for total yield uncertainty per channel
    total_variance_list += [np.asarray([0]) for _ in range(n_channels)]
    total_variance = ak.from_iter(total_variance_list)

    # loop over parameters to sum up total variance
    # first do the diagonal of the correlation matrix
    for i_par in range(model.config.npars):
        symmetric_uncertainty = (up_variations[i_par] - down_variations[i_par]) / 2
        total_variance = total_variance + symmetric_uncertainty ** 2

    labels = model.config.par_names()
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
                    continue  # two different staterrors are orthogonal, no contribution
                sym_unc_i = (up_variations[i_par] - down_variations[i_par]) / 2
                sym_unc_j = (up_variations[j_par] - down_variations[j_par]) / 2
                # factor of two below is there since loop is only over half the matrix
                total_variance = total_variance + 2 * (corr * sym_unc_i * sym_unc_j)

    # convert to standard deviations per bin and per channel
    total_stdev_per_bin = np.sqrt(total_variance[:n_channels])
    total_stdev_per_channel = ak.flatten(np.sqrt(total_variance[n_channels:]))
    log.debug(f"total stdev is {total_stdev_per_bin}")
    log.debug(f"total stdev per channel is {total_stdev_per_channel}")

    # convert to lists
    total_stdev_per_bin = ak.to_list(total_stdev_per_bin)
    total_stdev_per_channel = ak.to_list(total_stdev_per_channel)

    # save to cache
    _YIELD_STDEV_CACHE.update(
        {
            (model, tuple(parameters), tuple(uncertainty), corr_mat.data.tobytes()): (
                total_stdev_per_bin,
                total_stdev_per_channel,
            )
        }
    )

    return total_stdev_per_bin, total_stdev_per_channel


def prediction(
    model: pyhf.pdf.Model,
    fit_results: Optional[FitResults] = None,
    label: Optional[str] = None,
) -> ModelPrediction:
    """Returns model prediction, including model yields and uncertainties.

    If the optional fit result is not provided, the pre-fit Asimov yields and
    uncertainties are calculated. If the fit result is provided, the best-fit parameter
    values, uncertainties, and the parameter correlations are used to obtain the post-
    fit model and its associated uncertainties.

    Args:
        model (pyhf.pdf.Model): model to evaluate yield prediction for
        fit_results (Optional[FitResults], optional): parameter configuration to use,
            includes best-fit settings and uncertainties, as well as correlation matrix,
            defaults to None (then the pre-fit configuration is used)
        label (Optional[str], optional): label to include in model prediction, defaults
            to None (then will use "pre-fit" if fit results are not included, and "post-
            fit" otherwise)

    Returns:
        ModelPrediction: model, yields and uncertainties per bin and channel
    """
    if fit_results is not None:
        # fit results specified, so they are used
        param_values = fit_results.bestfit
        param_uncertainty = fit_results.uncertainty
        corr_mat = fit_results.corr_mat
        label = "post-fit" if label is None else label

    else:
        # no fit results specified, generate pre-fit parameter values, uncertainties,
        # and diagonal correlation matrix
        param_values = asimov_parameters(model)
        param_uncertainty = prefit_uncertainties(model)
        corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
        np.fill_diagonal(corr_mat, 1.0)
        label = "pre-fit" if label is None else label

    yields_combined = pyhf.tensorlib.to_numpy(
        model.main_model.expected_data(param_values, return_by_sample=True)
    )  # all channels concatenated

    # slice the yields into list of lists (of lists) where first index is channel,
    # second index is sample (and third index is bin)
    region_split_indices = _channel_boundary_indices(model)
    model_yields = [
        m.tolist() for m in np.split(yields_combined, region_split_indices, axis=1)
    ]

    # calculate the total standard deviation of the model prediction
    # indices: channel (and bin) for per-bin uncertainties, channel for per-channel
    total_stdev_model_bins, total_stdev_model_channels = yield_stdev(
        model, param_values, param_uncertainty, corr_mat
    )

    return ModelPrediction(
        model, model_yields, total_stdev_model_bins, total_stdev_model_channels, label
    )


def unconstrained_parameter_count(model: pyhf.pdf.Model) -> int:
    """Returns the number of unconstrained parameters in a model.

    The number is the sum of all independent parameters in a fit. A shapefactor that
    affects multiple bins enters the count once for each independent bin. Parameters
    that are set to constant are not included in the count.

    Args:
        model (pyhf.pdf.Model): model to count parameters for

    Returns:
        int: number of unconstrained parameters
    """
    n_pars = 0
    for parname in model.config.par_order:
        if (
            not model.config.param_set(parname).constrained
            and not model.config.param_set(parname).suggested_fixed
        ):
            n_pars += model.config.param_set(parname).n_parameters
    return n_pars


def _parameter_index(par_name: str, labels: Union[List[str], Tuple[str, ...]]) -> int:
    """Returns the position of a parameter with a given name in the list of parameters.

    Useful together with ``pyhf.pdf._ModelConfig.par_names`` to find the position of a
    parameter when the name is known. If the parameter is not found, logs an error and
    returns a default value of -1.

    Args:
        par_name (str): name of parameter to find in list
        labels (Union[List[str], Tuple[str, ...]]): list or tuple with all parameter
            names in the model

    Returns:
        int: index of parameter
    """
    par_index = next((i for i, label in enumerate(labels) if label == par_name), -1)
    if par_index == -1:
        log.error(f"parameter {par_name} not found in model")
    return par_index


def _strip_auxdata(model: pyhf.pdf.Model, data: List[float]) -> List[float]:
    """Always returns observed yields, no matter whether data includes auxdata.

    Args:
        model (pyhf.pdf.Model): model to which data corresponds to
        data (List[float]): data, either including auxdata which is then stripped off or
            only observed yields

    Returns:
        List[float]: observed data yields
    """
    n_bins_total = sum(model.config.channel_nbins.values())
    if len(data) != n_bins_total:
        # strip auxdata, only observed yields are needed
        data = data[:n_bins_total]
    return data


def _data_per_channel(model: pyhf.pdf.Model, data: List[float]) -> List[List[float]]:
    """Returns data split per channel, and strips off auxiliary data if included.

    Args:
        model (pyhf.pdf.Model): model to which data corresponds to
        data (List[float]): data (not split by channel), can either include auxdata
            which is then stripped off, or only observed yields

    Returns:
        List[List[float]]: data per channel and per bin
    """
    # strip off auxiliary data
    data_combined = _strip_auxdata(model, data)
    channel_split_indices = _channel_boundary_indices(model)
    # data is indexed by channel (and bin)
    data_yields = [d.tolist() for d in np.split(data_combined, channel_split_indices)]
    return data_yields


def _filter_channels(
    model: pyhf.pdf.Model, channels: Optional[Union[str, List[str]]]
) -> List[str]:
    """Returns a list of channels in a model after applying filtering.

    Args:
        model (pyhf.pdf.Model): model from which to extract channels
        channels (Optional[Union[str, List[str]]]): name of channel or list of channels
            to filter, only including those channels provided via this argument in the
            return of the function

    Returns:
        List[str]: list of channels after filtering
    """
    # channels included in model
    filtered_channels = model.config.channels
    # if one or more custom channels are provided, only include those
    if channels is not None:
        if isinstance(channels, str):
            channels = [channels]  # ensure list
        # only include if channel exists in model
        filtered_channels = [ch for ch in channels if ch in model.config.channels]

    if filtered_channels == []:
        log.warning(
            f"channel(s) {channels} not found in model, available channel(s): "
            f"{model.config.channels}"
        )

    return filtered_channels
