import logging
import pathlib
from typing import Any, Dict, Optional, Union

import numpy as np
import pyhf

from . import configuration
from . import fit
from . import model_utils
from . import template_builder

log = logging.getLogger(__name__)


def data_MC(
    config: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path],
    spec: Dict[str, Any],
    fit_results: Optional[fit.FitResults] = None,
    method: str = "matplotlib",
) -> None:
    """Draws a data/MC histogram.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        spec (Dict[str, Any]): ``pyhf`` workspace specification
        fit_results (Optional[fit.FitResults]): parameter configuration to use for plot,
            includes best-fit settings and uncertainties, as well as correlation matrix,
            defaults to None (then the pre-fit configuration is drawn)
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

    if fit_results:
        # fit results specified, draw a post-fit plot with them applied
        prefit = False
        param_values = fit_results.bestfit
        param_uncertainty = fit_results.uncertainty
        corr_mat = fit_results.corr_mat

    else:
        # no fit results specified, draw a pre-fit plot
        prefit = True
        # use pre-fit parameter values and uncertainties, and diagonal correlation matrix
        param_values, param_uncertainty = model_utils.get_asimov_parameters(model)
        corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
        np.fill_diagonal(corr_mat, 1.0)

    yields_combined = model.main_model.expected_data(
        param_values, return_by_sample=True
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
    total_stdev_model = model_utils.calculate_stdev(
        model, param_values, param_uncertainty, corr_mat
    )

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
