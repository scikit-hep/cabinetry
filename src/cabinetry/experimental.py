import logging
import pathlib
from typing import Any, Dict, Union

import awkward1 as ak
import numpy as np
import pyhf

from . import configuration
from . import model_utils
from . import template_builder

log = logging.getLogger(__name__)


def calculate_stdev(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
) -> ak.highlevel.Array:
    """Calculate the yield standard deviation of a model.

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
    region_split_indices = [
        model.config.channel_nbins[chan] for chan in model.config.channels
    ][
        :-1
    ]  # last index dropped since no extra split is needed after the last bin

    # the lists up_variations and down_variations will contain the model distributions
    # with all parameters varied individually within uncertainties
    # indices: variation, channel, bin
    up_variations = []
    down_variations = []

    # get the model for every parameter varied up and down
    # within the respective uncertainties
    for i_par in range(model.config.npars):
        # best-fit parameter values, but one parameter varied within uncertainties
        up_pars = parameters.copy()
        up_pars[i_par] += uncertainty[i_par]
        down_pars = parameters.copy()
        down_pars[i_par] -= uncertainty[i_par]

        # total model distribution with this parameter varied up
        up_combined = model.expected_data(up_pars, include_auxdata=False)
        up_yields = np.split(up_combined, region_split_indices)
        up_variations.append(up_yields)

        # total model distribution with this parameter varied down
        down_combined = model.expected_data(down_pars, include_auxdata=False)
        down_yields = np.split(down_combined, region_split_indices)
        down_variations.append(down_yields)

    # convert to awkward arrays for further processing
    up_variations = ak.from_iter(up_variations)
    down_variations = ak.from_iter(down_variations)

    # total variance, indices are: channel, bins
    total_variance_list = [
        np.zeros(shape=(model.config.channel_nbins[chan]))
        for chan in model.config.channels
    ]  # list of arrays, each array has as many entries as there are bins
    total_variance = ak.from_iter(total_variance_list)

    # loop over parameters to sum up total variance
    # first do the diagonal of the correlation matrix
    for i_par in range(model.config.npars):
        symmetric_uncertainty = (up_variations[i_par] - down_variations[i_par]) / 2
        total_variance = total_variance + symmetric_uncertainty ** 2

    # off-diagonal contributions
    # could optimize here: no need to loop over staterrors, since they
    # contribute nothing (bin effects are orthogonal)
    for i_par in range(model.config.npars):
        for j_par in range(model.config.npars):
            if i_par == j_par:
                continue
            corr = corr_mat[i_par, j_par]
            if corr == 0:
                continue
            sym_unc_i = (up_variations[i_par] - down_variations[i_par]) / 2
            sym_unc_j = (up_variations[j_par] - down_variations[j_par]) / 2
            total_variance = total_variance + (corr * sym_unc_i * sym_unc_j)

    # convert to standard deviation
    total_stdev = np.sqrt(total_variance)
    log.debug(f"total stdev is {total_stdev}")
    return total_stdev


def data_MC(
    config: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path],
    spec: Dict[str, Any],
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
    prefit: bool = True,
    method: str = "matplotlib",
) -> None:
    """Draws a data/MC histogram.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        spec (Dict[str, Any]): ``pyhf`` workspace specification
        bestfit (np.ndarray): best-fit parameter values
        uncertainty (np.ndarray): parameter uncertainties
        corr_mat (np.ndarray): correlation matrix
        prefit (bool, optional): draws the pre-fit model if True, else post-fit,
            defaults to True
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    # relies on https://github.com/scikit-hep/pyhf/pull/731 for return_by_sample
    workspace = pyhf.Workspace(spec)
    model = workspace.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        }
    )

    asimov_pars, pre_fit_unc = model_utils.get_asimov_parameters(model)

    if prefit:
        # override post-fit quantities by pre-fit settings
        bestfit = np.asarray(
            asimov_pars
        )  # the np.asarray is due to https://github.com/scikit-hep/pyhf/issues/1027
        uncertainty = np.asarray(pre_fit_unc)
        corr_mat = np.zeros(shape=(len(bestfit), len(bestfit)))
        np.fill_diagonal(corr_mat, 1.0)

    yields_combined = model.main_model.expected_data(
        bestfit, return_by_sample=True
    )  # all channels concatenated
    data_combined = workspace.data(model, with_aux=False)

    # slice the yields into an array where the first index is the channel,
    # and the second index is the sample
    region_split_indices = [
        model.config.channel_nbins[chan] for chan in model.config.channels
    ][:-1]
    model_yields = np.split(yields_combined, region_split_indices, axis=1)
    data = np.split(data_combined, region_split_indices)  # data just indexed by channel

    # calculate the total standard deviation of the model prediction, index: channel
    total_stdev_model = calculate_stdev(model, bestfit, uncertainty, corr_mat)

    for i_chan, channel_name in enumerate(
        model.config.channels
    ):  # process channel by channel
        histogram_dict_list = []  # one dict per region/channel

        # get the region dictionary from the config for binning / variable name
        region_dict = configuration.get_region_dict(config, channel_name)
        bins = template_builder._get_binning(region_dict)
        variable = region_dict["Variable"]

        for i_sam, sample_name in enumerate(model.config.samples):
            histogram_dict_list.append(
                {
                    "label": sample_name,
                    "isData": False,
                    "hist": {
                        "bins": bins,
                        "yields": model_yields[i_chan][i_sam],
                        "stdev": np.zeros_like(model_yields[i_chan][i_sam]),
                    },
                    "variable": variable,
                }
            )

        # add data sample
        histogram_dict_list.append(
            {
                "label": "Data",
                "isData": True,
                "hist": {
                    "bins": bins,
                    "yields": data[i_chan],
                    "stdev": np.sqrt(data[i_chan]),
                },
                "variable": variable,
            }
        )

        if method == "matplotlib":
            from .contrib import matplotlib_visualize

            if prefit:
                figure_path = pathlib.Path(figure_folder) / (
                    channel_name + "_prefit.pdf"
                )
            else:
                figure_path = pathlib.Path(figure_folder) / (
                    channel_name + "_postfit.pdf"
                )
            matplotlib_visualize.data_MC(
                histogram_dict_list,
                figure_path,
                total_unc=np.asarray(total_stdev_model[i_chan]),
            )
        else:
            raise NotImplementedError(f"unknown backend: {method}")
