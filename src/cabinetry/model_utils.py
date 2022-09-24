"""Provides utilities for pyhf models."""

from collections import defaultdict
import json
import logging
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Tuple, Union

try:
    # use awkward v2
    import awkward._v2 as ak
except ModuleNotFoundError:  # pragma: no cover
    # fallback if the _v2 submodule disappears after the full v2 release
    import awkward as ak  # pragma: no cover
import numpy as np
import pyhf

from cabinetry.fit.results_containers import FitResults


log = logging.getLogger(__name__)


# cache holding results from yield uncertainty calculations
_YIELD_STDEV_CACHE: Dict[Any, Tuple[List[List[List[float]]], List[List[float]]]] = {}


class ModelPrediction(NamedTuple):
    """Model prediction with yields and total uncertainties per bin and channel.

    Args:
        model (pyhf.pdf.Model): model to which prediction corresponds to
        model_yields (List[List[List[float]]]): yields per sample, channel and bin,
            indices: channel, sample, bin
        total_stdev_model_bins (List[List[List[float]]]): total yield uncertainty per
            channel, sample and bin, indices: channel, sample, bin (last sample: sum
            over samples)
        total_stdev_model_channels (List[List[float]]): total yield uncertainty per
            channel and sample, indices: channel, sample (last sample: sum over samples)
        label (str): label for the prediction, e.g. "pre-fit" or "post-fit"
    """

    model: pyhf.pdf.Model
    model_yields: List[List[List[float]]]
    total_stdev_model_bins: List[List[List[float]]]
    total_stdev_model_channels: List[List[float]]
    label: str


def model_and_data(
    spec: Dict[str, Any], *, asimov: bool = False, include_auxdata: bool = True
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


def asimov_data(
    model: pyhf.Model,
    *,
    fit_results: Optional[FitResults] = None,
    poi_name: Optional[str] = None,
    poi_value: Optional[float] = None,
    include_auxdata: bool = True,
) -> List[float]:
    """Returns the Asimov dataset (optionally with auxdata) for a model.

    Initial parameter settings for normalization factors in the workspace are treated as
    the default settings for that parameter. Fitting the Asimov dataset will recover
    these initial settings as the maximum likelihood estimate for normalization factors.
    Initial settings for other modifiers are ignored. If the fit_results keyword
    argument is used, the Asimov dataset is built to recover the fit results given when
    fitted again.

    Args:
        model (pyhf.Model): the model from which to construct the dataset
        fit_results (Optional[FitResults], optional): parameter configuration to use
            when building the Asimov dataset (using the best-fit result), defaults to
            None (then a pre-fit Asimov dataset is built)
        poi_name (Optional[str], optional): name of parameter to set to a custom value
            via poi_value, defaults to None (use POI specified in workspace)
        poi_value (Optional[float], optional): custom value to set POI specified via
            poi_name to, defaults to None (no custom value set)
        include_auxdata (bool, optional): whether to also return auxdata, defaults to
            True

    Returns:
        List[float]: the Asimov dataset
    """
    if fit_results is None:
        # pre-fit Asimov by default
        parameters = asimov_parameters(model)
    else:
        # Asimov data for given fit result (copy to not modify original fit_results)
        parameters = fit_results.bestfit.copy()

    if poi_value is not None:
        # set POI to custom value, using parameter specified in model or given in kwarg
        poi_index = _poi_index(model, poi_name=poi_name)
        if poi_index is None:
            raise ValueError(
                "no POI specified in model, use the poi_name argument to set POI name"
            )
        parameters[poi_index] = poi_value

    asimov_data = pyhf.tensorlib.tolist(
        model.expected_data(parameters, include_auxdata=include_auxdata)
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
            and not model.config.param_set(parameter).suggested_fixed_as_bool
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


def _hashable_model_key(
    model: pyhf.pdf.Model,
) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    """Compute a hashable representation of the values that uniquely identify a model.

    The ``pyhf.pdf.Model`` type is already hashable, but it uses the ``__hash__``
    inherited from ``object``, so a copy of a model has a distinct hash. The key
    returned by this function instead will hash to the same value for copies, but differ
    when the model represents a different likelihood.

    Note: The key returned here considers only the spec and interpolation codes. All
    other model configuration options leave it unchanged (e.g. ``poi_name``, overriding
    parameter bounds, etc.).

    Args:
        model (pyhf.model.Model): model to generate a key for

    Returns:
        Tuple[str, Tuple[Tuple[str, str], ...]]: a key that identifies the model
        by its spec and interpcodes
    """
    interpcodes = []
    for mod_type in sorted(model.config.modifier_settings.keys()):
        code = model.config.modifier_settings[mod_type]["interpcode"]
        interpcodes.append((mod_type, code))
    # sort since different orderings result in equivalent models,
    # but distinct strings
    spec_str = json.dumps(model.spec, sort_keys=True)
    return (spec_str, tuple(interpcodes))


def yield_stdev(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
) -> Tuple[List[List[List[float]]], List[List[float]]]:
    """Calculates symmetrized model yield standard deviation per channel / sample / bin.

    Returns both the uncertainties per bin (in a list of channels and samples), and the
    uncertainty of the total yield per channel (again, for a list of channels and
    samples). To calculate the uncertainties for the total yield per channel, the
    function internally treats the sum of yields per channel like another channel with
    one bin. Similarly, the sum over samples is treated as another sample. The results
    of this function are cached to speed up subsequent calls with the same arguments.

    Args:
        model (pyhf.pdf.Model): the model for which to calculate the standard deviations
            for all bins
        parameters (np.ndarray): central values of model parameters
        uncertainty (np.ndarray): uncertainty of model parameters
        corr_mat (np.ndarray): correlation matrix

    Returns:
        Tuple[List[List[List[float]]], List[List[float]]]:
            - list of channels, each channel is a list of samples, and each sample a
              list of standard deviations per bin (the last sample corresponds to a sum
              over all samples)
            - list of standard deviations per channel, each channel is a list containing
              the standard deviations per sample (the last sample corresponds to a sum
              over all samples)
    """
    # check whether results are already stored in cache
    cached_results = _YIELD_STDEV_CACHE.get(
        (
            _hashable_model_key(model),
            tuple(parameters),
            tuple(uncertainty),
            corr_mat.data.tobytes(),
        ),
        None,
    )
    if cached_results is not None:
        # return results from cache
        return cached_results

    # the lists up_variations and down_variations will contain the model distributions
    # with all parameters varied individually within uncertainties
    # indices: variation, channel, sample, bin
    # following the channels contained in the model, there are additional entries with
    # yields summed per channel (internally treated like additional channels) to get the
    # per-channel uncertainties
    # in the same way, the total model prediction is following the list of individual
    # samples (and then treated like an additional sample)
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

        # model distribution per sample with this parameter varied up
        up_comb = pyhf.tensorlib.to_numpy(
            model.main_model.expected_data(up_pars, return_by_sample=True)
        )
        # attach another entry with the total model prediction (sum over all samples)
        # indices: sample, bin
        up_comb = np.vstack((up_comb, np.sum(up_comb, axis=0)))
        # turn into list of channels (keep all samples, select correct bins per channel)
        # indices: channel, sample, bin
        up_yields = [
            up_comb[:, model.config.channel_slices[ch]] for ch in model.config.channels
        ]
        # append list of yields summed per channel
        up_yields += [
            np.asarray(np.sum(chan_yields, axis=-1, keepdims=True))
            for chan_yields in up_yields
        ]
        # indices: variation, channel, sample, bin
        up_variations.append(up_yields)

        # model distribution per sample with this parameter varied down
        down_comb = pyhf.tensorlib.to_numpy(
            model.main_model.expected_data(down_pars, return_by_sample=True)
        )
        # add total prediction (sum over samples)
        down_comb = np.vstack((down_comb, np.sum(down_comb, axis=0)))
        # turn into list of channels
        down_yields = [
            down_comb[:, model.config.channel_slices[ch]]
            for ch in model.config.channels
        ]
        # append list of yields summed per channel
        down_yields += [
            np.asarray(np.sum(chan_yields, axis=-1, keepdims=True))
            for chan_yields in down_yields
        ]
        down_variations.append(down_yields)

    # convert to awkward arrays for further processing
    up_variations_ak = ak.from_iter(up_variations)
    down_variations_ak = ak.from_iter(down_variations)

    # calculate symmetric uncertainties for all components
    # indices: variation, channel (last entries sums), sample (last entry sum), bin
    sym_uncs = (up_variations_ak - down_variations_ak) / 2

    # calculate total variance, indexed by channel, sample, bin (per-channel numbers act
    # like additional channels with one bin each)
    if np.count_nonzero(corr_mat - np.diagflat(np.ones_like(parameters))) == 0:
        # no off-diagonal contributions from correlation matrix (e.g. pre-fit)
        total_variance = ak.sum(np.power(sym_uncs, 2), axis=0)
    else:
        # full calculation including off-diagonal contributions
        # with v as vector of variations (each element contains yields under variation)
        # and M as correlation matrix, calculate variance as follows:
        # variance = sum_i sum_j v[i] * M[i, j] * v[j]
        # where the product between elements of v again is elementwise (multiplying bin
        # yields), and the final variance shape is the same as element of v (yield
        # uncertainties per bin, sample, channel)

        # possible optimizations that could be considered here:
        #   - skipping staterror-staterror terms for per-bin calculation (orthogonal)
        #   - taking advantage of correlation matrix symmetry
        #   - (optional) skipping combinations with correlations below threshold

        # calculate M[i, j] * v[j] first
        # indices: pars (i), pars (j), channel, sample, bin
        m_times_v = (
            corr_mat[..., np.newaxis, np.newaxis, np.newaxis]
            * sym_uncs[np.newaxis, ...]
        )
        # now multiply by v[i] as well, indices: pars(i), pars(j), channel, sample, bin
        v_times_m_times_v = sym_uncs[:, np.newaxis, ...] * m_times_v
        # finally perform sums over i and j, remaining indices: channel, sample, bin
        total_variance = ak.sum(ak.sum(v_times_m_times_v, axis=1), axis=0)

    # convert to standard deviations per bin and per channel
    n_channels = len(model.config.channels)
    # indices: (channel, sample, bin)
    total_stdev_per_bin = np.sqrt(total_variance[:n_channels])
    # indices: (channel, sample)
    total_stdev_per_channel = ak.sum(np.sqrt(total_variance[n_channels:]), axis=-1)
    # log total stdev per bin / channel (-1 index for sample sum)
    log.debug(f"total stdev is {total_stdev_per_bin[:, -1, :]}")
    log.debug(f"total stdev per channel is {total_stdev_per_channel[:, -1]}")

    # convert to lists
    total_stdev_per_bin = ak.to_list(total_stdev_per_bin)
    total_stdev_per_channel = ak.to_list(total_stdev_per_channel)

    # save to cache
    _YIELD_STDEV_CACHE.update(
        {
            (
                _hashable_model_key(model),
                tuple(parameters),
                tuple(uncertainty),
                corr_mat.data.tobytes(),
            ): (
                total_stdev_per_bin,
                total_stdev_per_channel,
            )
        }
    )

    return total_stdev_per_bin, total_stdev_per_channel


def prediction(
    model: pyhf.pdf.Model,
    *,
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
        if fit_results.labels != model.config.par_names:
            log.warning("parameter names in fit results and model do not match")
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
    model_yields = [
        yields_combined[:, model.config.channel_slices[ch]].tolist()
        for ch in model.config.channels
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
            and not model.config.param_set(parname).suggested_fixed_as_bool
        ):
            n_pars += model.config.param_set(parname).n_parameters
    return n_pars


def _parameter_index(
    par_name: str, labels: Union[List[str], Tuple[str, ...]]
) -> Optional[int]:
    """Returns the position of a parameter with a given name in the list of parameters.

    Useful together with ``pyhf.pdf._ModelConfig.par_names`` to find the position of a
    parameter when the name is known. If the parameter is not found, logs an error and
    returns a default value of None.

    Args:
        par_name (str): name of parameter to find in list
        labels (Union[List[str], Tuple[str, ...]]): list or tuple with all parameter
            names in the model

    Returns:
        Optional[int]: index of parameter, or None if parameter was not found
    """
    par_index = next((i for i, label in enumerate(labels) if label == par_name), None)
    if par_index is None:
        log.error(f"parameter {par_name} not found in model")
    return par_index


def _poi_index(
    model: pyhf.pdf.Model, *, poi_name: Optional[str] = None
) -> Optional[int]:
    """Returns the index of the POI specified in the argument or the model default.

    If a string is given as argument, this takes priority. Otherwise the POI from the
    model is used. If no POI is specified there either, logs an error and returns a
    default value of None.

    Args:
        model (pyhf.pdf.Model): model for which to find the POI index
        poi_name (Optional[str], optional): name of the POI, defaults to None

    Raises:
        ValueError: if the specified POI name cannot be found in the model

    Returns:
        Optional[int]: POI index, or None if no POI could be found
    """
    if poi_name is not None:
        # use POI given by kwarg if specified
        poi_index = _parameter_index(poi_name, model.config.par_names)
        if poi_index is None:
            raise ValueError(f"parameter {poi_name} not found in model")
    elif model.config.poi_index is not None:
        # use POI specified in model
        poi_index = model.config.poi_index
    else:
        log.error("could not find POI for model")
        poi_index = None

    return poi_index


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

    # data is indexed by channel (and bin)
    data_yields = [
        data_combined[model.config.channel_slices[ch]] for ch in model.config.channels
    ]
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


def match_fit_results(model: pyhf.pdf.Model, fit_results: FitResults) -> FitResults:
    """Matches results from a fit to a model by adding or removing parameters as needed.

    If the fit results contain parameters missing in the model, these parameters are not
    included in the returned fit results. If the fit results do not include parameters
    used in the model, they are added to the fit results. The best-fit value for such
    parameters are the Asimov values as returned by ``asimov_parameters`` (initial
    parameter settings for unconstrained parameters), and the associated uncertainties
    as given by ``prefit_uncertainties`` (zero uncertainty for unconstrained or fixed
    parameters). These parameters furthermore are assumed to have no correlation with
    any other parameters. If required, parameters are re-ordered to match the target
    model.

    Args:
        model (pyhf.pdf.Model): model to match fit results to
        fit_results (FitResults): fit results to be updated in order to match model

    Returns:
        FitResults: fit results matching the model
    """
    # start with the assumption that the provided fit results contain no relevant
    # information for the target model at all, and initialize everything accordingly
    # then inject information contained in provided fit results and return the new
    # fit results matching the target model at the end

    bestfit = asimov_parameters(model)  # Asimov parameter values for target model
    uncertainty = prefit_uncertainties(model)  # pre-fit uncertainties for target model
    labels = model.config.par_names  # labels for target model

    # indices of parameters in current fit results, or None if they are missing
    indices_for_corr: List[Optional[int]] = [None] * len(labels)

    # loop over all required parameters
    for target_idx, target_label in enumerate(labels):
        # if parameters are missing in fit results, no further action is needed here
        # as all relevant objects have been initialized with that assumption
        if target_label in fit_results.labels:
            # fit results contain parameter, find its position
            idx_in_fit_results = fit_results.labels.index(target_label)
            # override pre-fit values by actual fit results for parameters
            bestfit[target_idx] = fit_results.bestfit[idx_in_fit_results]
            uncertainty[target_idx] = fit_results.uncertainty[idx_in_fit_results]
            # update indices for correlation matrix reconstruction
            indices_for_corr[target_idx] = idx_in_fit_results

    # re-build correlation matrix: start with diagonal matrix (assuming fit results do
    # not contain relevant info), and then insert values provided in fit results
    corr_mat = np.diagflat(np.ones_like(labels, dtype=float))
    for i_target, i_orig in enumerate(indices_for_corr):
        for j_target, j_orig in enumerate(indices_for_corr):
            # i_target and j_target are positions in matched correlation matrix
            # i_orig and j_orig are positions in old matrix, None if missing from there
            if i_orig is not None and j_orig is not None:
                # if i_orig or j_orig are None, one of the parameters are not part of
                # original fit result, and no update to correlation matrix is needed
                corr_mat[i_target][j_target] = fit_results.corr_mat[i_orig][j_orig]

    fit_results_matched = FitResults(
        np.asarray(bestfit),
        np.asarray(uncertainty),
        labels,
        corr_mat,
        fit_results.best_twice_nll,
        fit_results.goodness_of_fit,
    )
    return fit_results_matched


def _modifier_map(
    model: pyhf.pdf.Model,
) -> DefaultDict[Tuple[str, str, str], List[str]]:
    """Creates a map for modifier lists per (channel, sample, parameter).

    Args:
        model (pyhf.pdf.Model): model for which to create the map

    Returns:
        Dict[Tuple[str, str, str], List[str]]: map to extract modifier lists for each
        (channel, sample, parameter).
    """
    modifier_map = defaultdict(list)
    for channel in model.spec["channels"]:
        for sample in channel["samples"]:
            for modifier in sample["modifiers"]:
                modifier_map[
                    (channel["name"], sample["name"], modifier["name"])
                ].append(modifier["type"])
    return modifier_map
