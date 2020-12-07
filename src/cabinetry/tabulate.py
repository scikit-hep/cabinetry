import logging
from typing import Any, Dict, List

import numpy as np
import pyhf
import tabulate


log = logging.getLogger(__name__)


def _header_name(channel_name: str, i_bin: int) -> str:
    """Constructs the header name for a column in a yield table.

    Args:
        channel_name (str): name of the channel (phase space region)
        i_bin (int): index of bin in channel

    Returns:
        str: the header name to be used for the column
    """
    if i_bin == 0:
        header_name = f"{channel_name}\nbin {i_bin+1}"
    else:
        header_name = f"\nbin {i_bin+1}"
    return header_name


def _yields(
    model: pyhf.pdf.Model,
    model_yields: List[np.ndarray],
    total_stdev_model: List[List[float]],
    data: List[np.ndarray],
) -> List[Dict[str, Any]]:
    """Outputs and returns a yield table with predicted and observed yields per bin.

    Args:
        model (pyhf.pdf.Model): the model which the table corresponds to
        model_yields (List[np.ndarray]): yields per channel, sample, and bin
        total_stdev_model (List[List[float]]): total model standard deviation per
            channel and bin
        data (List[np.ndarray]): data yield per channel and bin

    Returns:
        List[Dict[str, Any]]: yield table for use with the ``tabulate`` package
    """
    table = []  # table containing all yields

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
            total_dict.update(
                {
                    _header_name(channel_name, i_bin): f"{total_model[i_bin]:.2f} "
                    f"\u00B1 {total_stdev_model[i_chan][i_bin]:.2f}"
                }
            )
            data_dict.update(
                {_header_name(channel_name, i_bin): f"{data[i_chan][i_bin]:.2f}"}
            )
    table += [total_dict, data_dict]

    log.info(
        "yield table:\n"
        + tabulate.tabulate(table, headers="keys", tablefmt="fancy_grid")
    )

    return table
