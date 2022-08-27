"""Creates yield tables."""

import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import awkward as ak
import numpy as np
import pyhf
import tabulate

from cabinetry import model_utils


log = logging.getLogger(__name__)


def _header_name(channel_name: str, i_bin: int, *, unique: bool = True) -> str:
    """Constructs the header name for a column in a yield table.

    There are two modes: by default the names are unique (to be used as keys). With
    ``unique=False``, the channel names are skipped for bins beyond the first bin (for
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


def _save_tables(
    table_dict: Dict[str, List[Dict[str, Any]]],
    table_folder: pathlib.Path,
    table_label: str,
    table_format: str,
) -> None:
    """Saves yield tables in a specific format in a given folder.

    Newlines are removed from table headers for table formats that do not support them.

    Args:
        table_dict (Dict[str, List[Dict[str, Any]]]): dictionary with tables to save
        table_folder (pathlib.Path): path to the folder to save tables in
        table_label (str): label for tables to include in filenames
        table_format (str): format in which to save the tables
    """
    if table_format in ["plain", "simple", "tsv"]:
        save_suffix = "txt"
    elif table_format == "latex":
        save_suffix = "tex"
    else:
        save_suffix = table_format

    table_folder.mkdir(parents=True, exist_ok=True)

    for table_type in table_dict.keys():
        table_path = table_folder / f"{table_type}_{table_label}.{save_suffix}"

        if table_type == "yields_per_bin" and table_format in ["html", "latex", "tsv"]:
            # replace newlines in table headers for formats that do not support them
            # (newlines exist between channel name and bin number)
            for i_sample, sample in enumerate(table_dict[table_type]):
                table_dict[table_type][i_sample] = {
                    bin_name.replace("\n", ", "): sample_yield
                    for bin_name, sample_yield in sample.items()
                }

        table_str = (
            tabulate.tabulate(
                table_dict[table_type], headers="keys", tablefmt=table_format
            )
            + "\n"  # tabulate does not add a newline at the end
        )
        log.info(f"saving table as {table_path}")
        table_path.write_text(table_str)


def _yields_per_bin(
    model: pyhf.pdf.Model,
    model_yields: List[List[List[float]]],
    total_stdev_model: List[List[List[float]]],
    data: List[List[float]],
    channels: List[str],
    label: str,
) -> List[Dict[str, Any]]:
    """Outputs and returns a yield table with predicted and observed yields per bin.

    Args:
        model (pyhf.pdf.Model): the model which the table corresponds to
        model_yields (List[List[List[float]]]): yields per channel, sample, and bin
        total_stdev_model (List[List[List[float]]]): total model standard deviation per
            channel, sample and bin
        data (List[List[float]]): data yield per channel and bin
        channels (List[str]): names of channels to use
        label (str): label for model prediction to include in log

    Returns:
        List[Dict[str, Any]]: yield table for use with the ``tabulate`` package
    """
    table = []  # table containing all yields
    headers = {}  # headers with nicer formatting for output

    # indices of included channels
    channel_indices = [model.config.channels.index(ch) for ch in channels]

    # rows for each individual sample
    for i_sam, sample_name in enumerate(model.config.samples):
        sample_dict = {"sample": sample_name}  # one dict per sample
        for i_chan, channel_name in zip(channel_indices, channels):
            for i_bin in range(model.config.channel_nbins[channel_name]):
                sample_dict.update(
                    {
                        _header_name(
                            channel_name, i_bin
                        ): f"{model_yields[i_chan][i_sam][i_bin]:.2f} "
                        f"\u00B1 {total_stdev_model[i_chan][i_sam][i_bin]:.2f}"
                    }
                )
        table.append(sample_dict)

    # dicts for total model prediction and data
    total_dict = {"sample": "total"}
    data_dict = {"sample": "data"}
    for i_chan, channel_name in zip(channel_indices, channels):
        total_model = np.sum(model_yields[i_chan], axis=0)  # sum over samples
        for i_bin in range(model.config.channel_nbins[channel_name]):
            header_name = _header_name(channel_name, i_bin)
            headers.update(
                {header_name: _header_name(channel_name, i_bin, unique=False)}
            )
            total_dict.update(
                {
                    header_name: (
                        f"{total_model[i_bin]:.2f} "
                        f"\u00B1 {total_stdev_model[i_chan][-1][i_bin]:.2f}"
                    )
                }
            )
            data_dict.update({header_name: f"{data[i_chan][i_bin]:.2f}"})
    table += [total_dict, data_dict]

    log.info(
        f"yields per bin for {label} model prediction:\n"
        + tabulate.tabulate(table, headers=headers, tablefmt="fancy_grid")
    )
    return table


def _yields_per_channel(
    model: pyhf.pdf.Model,
    model_yields: List[List[float]],
    total_stdev_model: List[List[float]],
    data: List[float],
    channels: List[str],
    label: str,
) -> List[Dict[str, Any]]:
    """Outputs and returns a yield table with predicted and observed yields per channel.

    Args:
        model (pyhf.pdf.Model): the model which the table corresponds to
        model_yields (List[List[float]]): yields per channel and sample
        total_stdev_model (List[List[float]]): total model standard deviation per
            channel and sample
        data (List[float]): data yield per channel
        channels (List[str]): names of channels to use
        label (str): label for model prediction to include in log

    Returns:
        List[Dict[str, Any]]: yield table for use with the ``tabulate`` package
    """
    table = []  # table containing all yields

    # indices of included channels
    channel_indices = [model.config.channels.index(ch) for ch in channels]

    # rows for each individual sample
    for i_sam, sample_name in enumerate(model.config.samples):
        sample_dict = {"sample": sample_name}  # one dict per sample
        for i_chan, channel_name in zip(channel_indices, channels):
            sample_dict.update(
                {
                    channel_name: f"{model_yields[i_chan][i_sam]:.2f} "
                    f"\u00B1 {total_stdev_model[i_chan][i_sam]:.2f}"
                }
            )
        table.append(sample_dict)

    # dicts for total model prediction and data
    total_dict = {"sample": "total"}
    data_dict = {"sample": "data"}
    for i_chan, channel_name in zip(channel_indices, channels):
        total_model = np.sum(model_yields[i_chan], axis=0)  # sum over samples
        total_dict.update(
            {
                channel_name: f"{total_model:.2f} "
                f"\u00B1 {total_stdev_model[i_chan][-1]:.2f}"
            }
        )
        data_dict.update({channel_name: f"{data[i_chan]:.2f}"})
    table += [total_dict, data_dict]

    log.info(
        f"yields per channel for {label} model prediction:\n"
        + tabulate.tabulate(table, headers="keys", tablefmt="fancy_grid")
    )
    return table


def yields(
    model_prediction: model_utils.ModelPrediction,
    data: List[float],
    *,
    channels: Optional[Union[str, List[str]]] = None,
    per_bin: bool = True,
    per_channel: bool = False,
    table_folder: Union[str, pathlib.Path] = "tables",
    table_format: str = "simple",
    save_tables: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generates yield tables, showing model prediction and data.

    Channels can be filtered via the optional ``channels`` argument. Either yields per
    bin, or yields per channel, or both can be shown.

    Args:
        model_prediction (model_utils.ModelPrediction): model prediction to show in
            table
        data (List[float]): data to include in table, can either include auxdata (the
            auxdata is then stripped internally) or only observed yields
        channels (Optional[Union[str, List[str]]], optional): name of channel to show,
            or list of names to include, defaults to None (uses all channels)
        per_bin (bool, optional): whether to show a table with yields per bin, defaults
            to True
        per_channel (bool, optional): whether to show a table with yields per channel,
            defaults to False
        table_folder (Union[str, pathlib.Path], optional): path to the folder to save
            tables in, defaults to "tables"
        table_format (str, optional): format in which to save the tables, can be any of
            the formats ``tabulate`` supports (e.g. html, latex, plain, simple, tsv),
            defaults to "simple"
        save_tables (bool, optional): whether to save tables, defaults to True

    Returns:
        Dict[str, List[Dict[str, Any]]]: dictionary with yield tables for use with the
        ``tabulate`` package
    """
    table_dict: Dict[str, List[Dict[str, Any]]] = {}

    if not (per_bin or per_channel):
        log.warning(
            "requested neither yields per bin nor per channel, no table produced"
        )
        return table_dict

    # strip off auxdata (if needed) and obtain data indexed by channel (and bin)
    data_yields = model_utils._data_per_channel(model_prediction.model, data)

    # channels to include in table, with optional filtering applied
    filtered_channels = model_utils._filter_channels(model_prediction.model, channels)

    if filtered_channels == []:
        # nothing to include in tables, warning already raised via _filter_channels
        return table_dict

    if per_bin:
        # yield table with yields per bin
        per_bin_table = _yields_per_bin(
            model_prediction.model,
            model_prediction.model_yields,
            model_prediction.total_stdev_model_bins,
            data_yields,
            filtered_channels,
            model_prediction.label,
        )
        table_dict.update({"yields_per_bin": per_bin_table})

    if per_channel:
        # yields per channel
        model_yields_per_channel = np.sum(
            ak.from_iter(model_prediction.model_yields), axis=-1
        ).tolist()
        data_per_channel = [sum(d) for d in data_yields]
        per_channel_table = _yields_per_channel(
            model_prediction.model,
            model_yields_per_channel,
            model_prediction.total_stdev_model_channels,
            data_per_channel,
            filtered_channels,
            model_prediction.label,
        )
        table_dict.update({"yields_per_channel": per_channel_table})

    if save_tables:
        # save tables to file
        _save_tables(
            table_dict, pathlib.Path(table_folder), model_prediction.label, table_format
        )

    return table_dict
