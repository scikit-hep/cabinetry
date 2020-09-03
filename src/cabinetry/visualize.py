import glob
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import histo


log = logging.getLogger(__name__)


def _build_figure_name(region_name: str, is_prefit: bool) -> str:
    """construct a name for the file a figure is saved as

    Args:
        region_name (str): name of the region shown in the figure
        is_prefit (bool): whether the figure shows the pre- or post-fit model

    Returns:
        str: name of the file the figure should be saved to
    """
    figure_name = region_name.replace(" ", "-")
    if is_prefit:
        figure_name += "_" + "prefit"
    else:
        figure_name += "_" + "postfit"
    figure_name += ".pdf"
    return figure_name


def data_MC(
    config: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path],
    prefit: bool = True,
    method: str = "matplotlib",
) -> None:
    """draw a data/MC histogram, control whether it is pre- or postfit with a flag

    Args:
        config (Dict[str, Any]): cabinetry configuration
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        prefit (bool, optional): show the pre- or post-fit model, defaults to True
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
        NotImplementedError: when trying to visualize post-fit distributions, not supported yet
    """
    log.info("visualizing histogram")
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
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

                figure_path = pathlib.Path(figure_folder) / figure_name
                matplotlib_visualize.data_MC(histogram_dict_list, figure_path)
            else:
                raise NotImplementedError(f"unknown backend: {method}")
        else:
            raise NotImplementedError("only prefit implemented so far")


def correlation_matrix(
    corr_mat: np.ndarray,
    labels: List[str],
    figure_folder: Union[str, pathlib.Path],
    pruning_threshold: float = 0.0,
    method: str = "matplotlib",
) -> None:
    """plot a correlation matrix

    Args:
        corr_mat (np.ndarray): the correlation matrix to plot
        labels (List[str]): names of parameters in the correlation matrix
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        pruning_threshold (float, optional): minimum correlation for a parameter to
            have with any other parameters to not get pruned, defaults to 0.0
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    # create a matrix that's True if a correlation is below threshold, and True on the diagonal
    below_threshold = np.where(np.abs(corr_mat) <= pruning_threshold, True, False)
    np.fill_diagonal(below_threshold, True)
    # get indices of rows/columns where everything is below threshold
    delete_indices = np.where(np.all(below_threshold, axis=0))
    # delete rows and columns where all correlations are below threshold
    corr_mat = np.delete(
        np.delete(corr_mat, delete_indices, axis=1), delete_indices, axis=0
    )
    labels = np.delete(labels, delete_indices)

    figure_path = pathlib.Path(figure_folder) / "correlation_matrix.pdf"
    if method == "matplotlib":
        from cabinetry.contrib import matplotlib_visualize

        matplotlib_visualize.correlation_matrix(corr_mat, labels, figure_path)
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def pulls(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: List[str],
    figure_folder: Union[str, pathlib.Path],
    exclude_list: Optional[List[str]] = None,
    method: str = "matplotlib",
) -> None:
    """produce a pull plot of parameter results and uncertainties

    Args:
        bestfit (np.ndarray): best-fit results for parameters
        uncertainty (np.ndarray): parameter uncertainties
        labels (List[str]): parameter names
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        exclude_list (Optional[List[str]], optional): list of parameters to exclude from plot,
            defaults to None (nothing excluded)
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    figure_path = pathlib.Path(figure_folder) / "pulls.pdf"
    labels_np = np.asarray(labels)

    if exclude_list is None:
        exclude_list = []

    # exclude fixed parameters from pull plot
    exclude_list += [
        label for i_np, label in enumerate(labels_np) if uncertainty[i_np] == 0.0
    ]

    # exclude staterror parameters from pull plot (they are centered at 1)
    exclude_list += [label for label in labels_np if label[0:10] == "staterror_"]

    # filter out parameters
    mask = [True if label not in exclude_list else False for label in labels_np]
    bestfit = bestfit[mask]
    uncertainty = uncertainty[mask]
    labels_np = labels_np[mask]

    if method == "matplotlib":
        from cabinetry.contrib import matplotlib_visualize

        matplotlib_visualize.pulls(bestfit, uncertainty, labels_np, figure_path)
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def ranking(
    bestfit: np.ndarray,
    uncertainty: np.ndarray,
    labels: List[str],
    impact_prefit_up: np.ndarray,
    impact_prefit_down: np.ndarray,
    impact_postfit_up: np.ndarray,
    impact_postfit_down: np.ndarray,
    poi_index: int,
    figure_folder: Union[str, pathlib.Path],
    max_pars: Optional[int] = None,
    method: str = "matplotlib",
) -> None:
    """Produces a ranking plot showing the impact of parameters on the POI.

    Args:
        bestfit (np.ndarray): best-fit parameter results
        uncertainty (np.ndarray): parameter uncertainties
        labels (List[str]): parameter labels
        impact_prefit_up (np.ndarray): pre-fit impact in "up" direction per parameter
        impact_prefit_down (np.ndarray): pre-fit impact in "down" direction per parameter
        impact_postfit_up (np.ndarray): post-fit impact in "up" direction per parameter
        impact_postfit_down (np.ndarray): post-fit impact in "down" direction per parameter
        poi_index (int): index of POI in parameter list
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        max_pars (Optional[int], optional): number of parameters to include, defaults to None
            (which means all parameters are included)
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    figure_path = pathlib.Path(figure_folder) / "ranking.pdf"

    # remove the POI results from bestfit, uncertainty, labels
    bestfit = np.delete(bestfit, poi_index)
    uncertainty = np.delete(uncertainty, poi_index)
    labels = np.delete(labels, poi_index)

    # normalize staterrors - subtract 1
    # could also normalize width of staterrors here
    for i_par, label in enumerate(labels):
        if "staterror_" in label:
            bestfit[i_par] -= 1

    # sort parameters by decreasing average post-fit impact
    avg_postfit_impact = (np.abs(impact_postfit_up) + np.abs(impact_postfit_down)) / 2

    # get indices to sort by decreasing impact
    sorted_indices = np.argsort(avg_postfit_impact)[::-1]
    bestfit = bestfit[sorted_indices]
    uncertainty = uncertainty[sorted_indices]
    labels = labels[sorted_indices]
    impact_prefit_up = impact_prefit_up[sorted_indices]
    impact_prefit_down = impact_prefit_down[sorted_indices]
    impact_postfit_up = impact_postfit_up[sorted_indices]
    impact_postfit_down = impact_postfit_down[sorted_indices]

    if max_pars is not None:
        # only keep leading parameters in ranking
        bestfit = bestfit[:max_pars]
        uncertainty = uncertainty[:max_pars]
        labels = labels[:max_pars]
        impact_prefit_up = impact_prefit_up[:max_pars]
        impact_prefit_down = impact_prefit_down[:max_pars]
        impact_postfit_up = impact_postfit_up[:max_pars]
        impact_postfit_down = impact_postfit_down[:max_pars]

    if method == "matplotlib":
        from .contrib import matplotlib_visualize

        matplotlib_visualize.ranking(
            bestfit,
            uncertainty,
            labels,
            impact_prefit_up,
            impact_prefit_down,
            impact_postfit_up,
            impact_postfit_down,
            figure_path,
        )
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def templates(
    config: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path],
    method: str = "matplotlib",
) -> None:
    """Visualize template histograms for systematic variations.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        figure_folder (Union[str, pathlib.Path]): path to the folder to save figures in
        method (str, optional): what backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    log.info("visualizing systematics templates")
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    figure_folder = pathlib.Path(figure_folder) / "templates"

    # could do this via the route module instead
    for region in config["Regions"]:
        for sample in config["Samples"]:
            if sample.get("Data", False):
                # skip data
                continue

            for systematic in config["Systematics"]:
                histo_name = (
                    region["Name"]
                    + "_"
                    + sample["Name"]
                    + "_"
                    + systematic["Name"]
                    + "*_modified*"
                )
                # create a list of paths to histograms matching the pattern
                variation_paths = [
                    pathlib.Path(h_name)
                    for h_name in glob.glob(str(histogram_folder / histo_name))
                ]
                # only keep up/down variations, and sort alphabetically
                # (sorting to have consistent order, and simplified debugging)
                variation_paths = sorted(
                    [v for v in variation_paths if ("Up" in v.name or "Down" in v.name)]
                )

                if len(variation_paths) == 0:
                    # no associated templates (normalization systematics)
                    continue

                # extract nominal histogram
                nominal_histo = histo.Histogram.from_config(
                    histogram_folder,
                    region,
                    sample,
                    {"Name": "nominal"},
                )
                bins = nominal_histo.bins
                variable = region["Variable"]
                nominal = {"yields": nominal_histo.yields, "stdev": nominal_histo.stdev}

                # extract variation histograms if they exist
                up = {}
                down = {}
                for variation_path in variation_paths:
                    var_histo = histo.Histogram.from_path(variation_path)
                    var = {"yields": var_histo.yields, "stdev": var_histo.stdev}
                    if "Up" in variation_path.parts[-1]:
                        up.update(var)
                    else:
                        down.update(var)

                figure_name = (
                    region["Name"]
                    + "_"
                    + sample["Name"]
                    + "_"
                    + systematic["Name"]
                    + ".pdf"
                )
                figure_path = figure_folder / figure_name

                if method == "matplotlib":
                    from .contrib import matplotlib_visualize

                    matplotlib_visualize.templates(
                        nominal, up, down, bins, variable, figure_path
                    )

                else:
                    raise NotImplementedError(f"unknown backend: {method}")
