import logging
from typing import Any, Dict, List

import numpy as np
import pyhf
import tabulate


log = logging.getLogger(__name__)


def _header_name(channel_name: str, i_bin: int, unique: bool = True) -> str:
    """Constructs the header name for a column in a yield table.

    There are two modes: by default the names are unique (to be used as keys). With
    ``unique=False``, the region names are skipped for bins beyond the first bin (for
    less redundant output).

    Args:
        channel_name (str): name of the channel (phase space region)
        i_bin (int): index of bin in channel
        unique (bool, optional): whether to return a unique key, defaults to True

    Returns:
        str: the header name to be used for the column
    """
    if i_bin == 0 or unique:
        header_name = f"{channel_name}\nbin {i_bin+1}"
    else:
        header_name = f"\nbin {i_bin+1}"
    return header_name


def _yields_per_bin(
    model: pyhf.pdf.Model,
    model_yields: List[List[List[float]]],
    total_stdev_model: List[List[float]],
    data: List[List[float]],
) -> List[Dict[str, Any]]:
    """Outputs and returns a yield table with predicted and observed yields per bin.

    Args:
        model (pyhf.pdf.Model): the model which the table corresponds to
        model_yields (List[List[List[float]]]): yields per channel, sample, and bin
        total_stdev_model (List[List[float]]): total model standard deviation per
            channel and bin
        data (List[List[float]]): data yield per channel and bin

    Returns:
        List[Dict[str, Any]]: yield table for use with the ``tabulate`` package
    """
    table = []  # table containing all yields
    headers = {}  # headers with nicer formatting for output

    # rows for each individual sample
    for i_sam, sample_name in enumerate(model.config.samples):
        sample_dict = {"sample": sample_name}  # one dict per sample
        for i_chan, channel_name in enumerate(model.config.channels):
            for i_bin in range(model.config.channel_nbins[channel_name]):
                sample_dict.update(
                    {
                        _header_name(
                            channel_name, i_bin
                        ): f"{model_yields[i_chan][i_sam][i_bin]:.2f}"
                    }
                )
        table.append(sample_dict)

    # dicts for total model prediction and data
    total_dict = {"sample": "total"}
    data_dict = {"sample": "data"}
    for i_chan, channel_name in enumerate(model.config.channels):
        total_model = np.sum(model_yields[i_chan], axis=0)  # sum over samples
        for i_bin in range(model.config.channel_nbins[channel_name]):
            header_name = _header_name(channel_name, i_bin)
            headers.update(
                {header_name: _header_name(channel_name, i_bin, unique=False)}
            )
            total_dict.update(
                {
                    header_name: f"{total_model[i_bin]:.2f} "
                    f"\u00B1 {total_stdev_model[i_chan][i_bin]:.2f}"
                }
            )
            data_dict.update({header_name: f"{data[i_chan][i_bin]:.2f}"})
    table += [total_dict, data_dict]

    log.info(
        "yields per bin:\n"
        + tabulate.tabulate(
            table,
            headers=headers,
            tablefmt="fancy_grid",
        )
    )
    return table


def _yields_per_channel(
    model: pyhf.pdf.Model,
    model_yields: List[List[float]],
    total_stdev_model: List[float],
    data: List[float],
) -> List[Dict[str, Any]]:
    """Outputs and returns a yield table with predicted and observed yields per channel.

    Args:
        model (pyhf.pdf.Model): the model which the table corresponds to
        model_yields (List[List[float]]): yields per channel and sample
        total_stdev_model (List[float]): total model standard deviation per channel
        data (List[float]): data yield per channel

    Returns:
        List[Dict[str, Any]]: yield table for use with the ``tabulate`` package
    """
    table = []  # table containing all yields

    # rows for each individual sample
    for i_sam, sample_name in enumerate(model.config.samples):
        sample_dict = {"sample": sample_name}  # one dict per sample
        for i_chan, channel_name in enumerate(model.config.channels):
            sample_dict.update({channel_name: f"{model_yields[i_chan][i_sam]:.2f}"})
        table.append(sample_dict)

    # dicts for total model prediction and data
    total_dict = {"sample": "total"}
    data_dict = {"sample": "data"}
    for i_chan, channel_name in enumerate(model.config.channels):
        total_model = np.sum(model_yields[i_chan], axis=0)  # sum over samples
        total_dict.update(
            {
                channel_name: f"{total_model:.2f} "
                f"\u00B1 {total_stdev_model[i_chan]:.2f}"
            }
        )
        data_dict.update({channel_name: f"{data[i_chan]:.2f}"})
    table += [total_dict, data_dict]

    log.info(
        "yields per channel:\n"
        + tabulate.tabulate(
            table,
            headers="keys",
            tablefmt="fancy_grid",
        )
    )
    return table
