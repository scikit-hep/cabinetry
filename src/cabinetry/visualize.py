"""
need to save the histogram bins for visualization purposes still,
as well as cosmetics such as axis labels and region names
"""
import logging
from pathlib import Path
from typing import List

import numpy as np

from . import histo


log = logging.getLogger(__name__)


def _build_figure_name(region, is_prefit):
    """construct a name for the file a figure is saved as

    Args:
        region (dict): the region shown in the figure
        is_prefit (bool): whether the figure shows the pre- or post-fit model

    Returns:
        str: name of the file the figure should be saved to
    """
    figure_name = region.replace(" ", "-")
    if is_prefit:
        figure_name += "_" + "prefit"
    else:
        figure_name += "_" + "postfit"
    figure_name += ".pdf"
    return figure_name


def data_MC(config, histogram_folder, figure_folder, prefit=True, method="matplotlib"):
    """draw a data/MC histogram, control whether it is pre- or postfit with a flag

    Args:
        config (dict): cabinetry configuration
        histogram_folder (str): path to the folder containing template histograms
        figure_folder (str): path to the folder to save figures in
        prefit (bool, optional): show the pre- or post-fit model, defaults to True
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
        NotImplementedError: when trying to visualize post-fit distributions, not supported yet
    """
    log.info("visualizing histogram")
    for region in config["Regions"]:
        histogram_dict_list = []
        for sample in config["Samples"]:
            for systematic in [{"Name": "nominal"}]:
                is_data = sample.get("Data", False)
                histogram = histo.Histogram.from_config(
                    histogram_folder, region, sample, systematic, modified=True
                )
                histogram_dict_list.append(
                    {
                        "label": sample["Name"],
                        "isData": is_data,
                        "hist": {
                            "bins": histogram.bins,
                            "yields": histogram.yields,
                            "stdev": histogram.stdev,
                        },
                        "variable": region["Variable"],
                    }
                )

        figure_name = _build_figure_name(region["Name"], prefit)

        if prefit:
            if method == "matplotlib":
                from cabinetry.contrib import matplotlib_visualize

                figure_path = Path(figure_folder) / figure_name
                matplotlib_visualize.data_MC(histogram_dict_list, figure_path)
            else:
                raise NotImplementedError(f"unknown backend {method}")
        else:
            raise NotImplementedError("only prefit implemented so far")


def correlation_matrix(
    corr_mat: np.ndarray, labels: List[str], figure_folder: str, method="matplotlib"
) -> None:
    """plot a correlation matrix

    Args:
        corr_mat (np.ndarray): the correlation matrix to plot
        labels (List[str]): names of parameters in the correlation matrix
        figure_folder (str): path to the folder to save figures in
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    figure_path = Path(figure_folder) / "correlation_matrix.pdf"
    if method == "matplotlib":
        from cabinetry.contrib import matplotlib_visualize

        matplotlib_visualize.correlation_matrix(corr_mat, labels, figure_path)
    else:
        raise NotImplementedError(f"unknown backend {method}")
