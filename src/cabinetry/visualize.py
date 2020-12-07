import glob
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import configuration
from . import fit
from . import histo
from . import model_utils
from . import tabulate
from . import template_builder


log = logging.getLogger(__name__)


def _build_figure_name(region_name: str, is_prefit: bool) -> str:
    """Constructs a file name for a figure.

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


def _total_yield_uncertainty(stdev_list: List[np.ndarray]) -> np.ndarray:
    """Calculates the absolute statistical uncertainty of a stack of MC.

    Args:
        stdev_list (List[np.ndarray]): list of absolute stat. uncertainty per sample

    Returns:
        np.array: absolute stat. uncertainty of stack of samples
    """
    tot_unc = np.sqrt(np.sum(np.power(stdev_list, 2), axis=0))
    return tot_unc


def data_MC_from_histograms(
    config: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path] = "figures",
    log_scale: Optional[bool] = None,
    method: str = "matplotlib",
) -> None:
    """Draws pre-fit data/MC histograms, using histograms created by cabinetry.

    The uncertainty band drawn includes only statistical uncertainties.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        log_scale (Optional[bool], optional): whether to use logarithmic vertical axis,
            defaults to None (automatically determine whether to use linear/log scale)
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    log.info("visualizing histogram")
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    for region in config["Regions"]:
        histogram_dict_list = []
        model_stdevs = []
        # loop over samples in reverse order, such that samples that appear first in the
        # config will appear at the top of the stack in the plot (they are plotted last)
        for sample in config["Samples"][::-1]:
            is_data = sample.get("Data", False)
            histogram = histo.Histogram.from_config(
                histogram_folder, region, sample, {"Name": "Nominal"}, modified=True
            )
            histogram_dict_list.append(
                {
                    "label": sample["Name"],
                    "isData": is_data,
                    "yields": histogram.yields,
                    "variable": region["Variable"],
                }
            )
            if not is_data:
                model_stdevs.append(histogram.stdev)

        figure_name = _build_figure_name(region["Name"], True)
        total_model_unc = _total_yield_uncertainty(model_stdevs)
        bin_edges = histogram.bins

        if method == "matplotlib":
            from .contrib import matplotlib_visualize

            figure_path = pathlib.Path(figure_folder) / figure_name
            matplotlib_visualize.data_MC(
                histogram_dict_list,
                total_model_unc,
                bin_edges,
                figure_path,
                log_scale=log_scale,
            )
        else:
            raise NotImplementedError(f"unknown backend: {method}")


def data_MC(
    config: Dict[str, Any],
    spec: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path] = "figures",
    fit_results: Optional[fit.FitResults] = None,
    log_scale: Optional[bool] = None,
    include_table: bool = True,
    method: str = "matplotlib",
) -> None:
    """Draws pre- and post-fit data/MC histograms from a ``pyhf`` workspace.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        spec (Dict[str, Any]): ``pyhf`` workspace specification
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        fit_results (Optional[fit.FitResults]): parameter configuration to use for plot,
            includes best-fit settings and uncertainties, as well as correlation matrix,
            defaults to None (then the pre-fit configuration is drawn)
        log_scale (Optional[bool], optional): whether to use logarithmic vertical axis,
            defaults to None (automatically determine whether to use linear/log scale)
        include_table (bool, optional): whether to also output a yield table, defaults
            to True
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    model, data_combined = model_utils.model_and_data(spec, with_aux=False)

    if fit_results is not None:
        # fit results specified, draw a post-fit plot with them applied
        prefit = False
        param_values = fit_results.bestfit
        param_uncertainty = fit_results.uncertainty
        corr_mat = fit_results.corr_mat

    else:
        # no fit results specified, draw a pre-fit plot
        prefit = True
        # use pre-fit parameter values, uncertainties, and diagonal correlation matrix
        param_values = model_utils.get_asimov_parameters(model)
        param_uncertainty = model_utils.get_prefit_uncertainties(model)
        corr_mat = np.zeros(shape=(len(param_values), len(param_values)))
        np.fill_diagonal(corr_mat, 1.0)

    yields_combined = model.main_model.expected_data(
        param_values, return_by_sample=True
    )  # all channels concatenated

    # slice the yields into an array where the first index is the channel,
    # and the second index is the sample
    region_split_indices = model_utils._get_channel_boundary_indices(model)
    model_yields = np.split(yields_combined, region_split_indices, axis=1)
    data = np.split(data_combined, region_split_indices)  # data just indexed by channel

    # calculate the total standard deviation of the model prediction, index: channel
    total_stdev_model = model_utils.calculate_stdev(
        model, param_values, param_uncertainty, corr_mat
    )

    if include_table:
        # show yield table
        if prefit:
            log.info("generating pre-fit yield table")
        else:
            log.info("generating post-fit yield table")
        tabulate._yields(model, model_yields, total_stdev_model, data)

    # process channel by channel
    for i_chan, channel_name in enumerate(model.config.channels):
        histogram_dict_list = []  # one dict per region/channel

        # get the region dictionary from the config for binning / variable name
        region_dict = configuration.get_region_dict(config, channel_name)
        bin_edges = template_builder._get_binning(region_dict)
        variable = region_dict["Variable"]

        for i_sam, sample_name in enumerate(model.config.samples):
            histogram_dict_list.append(
                {
                    "label": sample_name,
                    "isData": False,
                    "yields": model_yields[i_chan][i_sam],
                    "variable": variable,
                }
            )

        # add data sample
        histogram_dict_list.append(
            {
                "label": "Data",
                "isData": True,
                "yields": data[i_chan],
                "variable": variable,
            }
        )

        if method == "matplotlib":
            from .contrib import matplotlib_visualize

            if prefit:
                figure_path = pathlib.Path(figure_folder) / _build_figure_name(
                    channel_name, True
                )
            else:
                figure_path = pathlib.Path(figure_folder) / _build_figure_name(
                    channel_name, False
                )
            matplotlib_visualize.data_MC(
                histogram_dict_list,
                np.asarray(total_stdev_model[i_chan]),
                bin_edges,
                figure_path,
                log_scale=log_scale,
            )
        else:
            raise NotImplementedError(f"unknown backend: {method}")


def correlation_matrix(
    fit_results: fit.FitResults,
    figure_folder: Union[str, pathlib.Path] = "figures",
    pruning_threshold: float = 0.0,
    method: str = "matplotlib",
) -> None:
    """Draws a correlation matrix.

    Args:
        fit_results (fit.FitResults): fit results, including correlation matrix and
            parameter labels
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        pruning_threshold (float, optional): minimum correlation for a parameter to
            have with any other parameters to not get pruned, defaults to 0.0
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    # create a matrix that is True if a correlation is below threshold, and True on the
    # diagonal
    below_threshold = np.where(
        np.abs(fit_results.corr_mat) < pruning_threshold, True, False
    )
    np.fill_diagonal(below_threshold, True)
    # get list of booleans specifying if everything in rows/columns is below threshold
    all_below_threshold = np.all(below_threshold, axis=0)
    # get list of booleans specifying if rows/columns correspond to fixed parameter
    # (0 correlations)
    fixed_parameter = np.all(np.equal(fit_results.corr_mat, 0.0), axis=0)
    # get indices of rows/columns where everything is below threshold, or the parameter
    # is fixed
    delete_indices = np.where(np.logical_or(all_below_threshold, fixed_parameter))
    # delete rows and columns where all correlations are below threshold / parameter is
    # fixed
    corr_mat = np.delete(
        np.delete(fit_results.corr_mat, delete_indices, axis=1), delete_indices, axis=0
    )
    labels = np.delete(fit_results.labels, delete_indices)

    figure_path = pathlib.Path(figure_folder) / "correlation_matrix.pdf"
    if method == "matplotlib":
        from .contrib import matplotlib_visualize

        matplotlib_visualize.correlation_matrix(corr_mat, labels, figure_path)
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def pulls(
    fit_results: fit.FitResults,
    figure_folder: Union[str, pathlib.Path] = "figures",
    exclude_list: Optional[List[str]] = None,
    method: str = "matplotlib",
) -> None:
    """Draws a pull plot of parameter results and uncertainties.

    Args:
        fit_results (fit.FitResults): fit results, including correlation matrix and
            parameter labels
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        exclude_list (Optional[List[str]], optional): list of parameters to exclude from
            plot, defaults to None (nothing excluded)
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    figure_path = pathlib.Path(figure_folder) / "pulls.pdf"
    labels_np = np.asarray(fit_results.labels)

    if exclude_list is None:
        exclude_list = []

    # exclude fixed parameters from pull plot
    exclude_list += [
        label
        for i_np, label in enumerate(labels_np)
        if fit_results.uncertainty[i_np] == 0.0
    ]

    # exclude staterror parameters from pull plot (they are centered at 1)
    exclude_list += [label for label in labels_np if label[0:10] == "staterror_"]

    # filter out parameters
    mask = [True if label not in exclude_list else False for label in labels_np]
    bestfit = fit_results.bestfit[mask]
    uncertainty = fit_results.uncertainty[mask]
    labels_np = labels_np[mask]

    if method == "matplotlib":
        from .contrib import matplotlib_visualize

        matplotlib_visualize.pulls(bestfit, uncertainty, labels_np, figure_path)
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def ranking(
    ranking_results: fit.RankingResults,
    figure_folder: Union[str, pathlib.Path] = "figures",
    max_pars: Optional[int] = None,
    method: str = "matplotlib",
) -> None:
    """Produces a ranking plot showing the impact of parameters on the POI.

    Args:
        ranking_results (fit.RankingResults): fit results, and pre- and post-fit impacts
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        max_pars (Optional[int], optional): number of parameters to include, defaults to
            None (which means all parameters are included)
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    figure_path = pathlib.Path(figure_folder) / "ranking.pdf"

    # sort parameters by decreasing average post-fit impact
    avg_postfit_impact = (
        np.abs(ranking_results.postfit_up) + np.abs(ranking_results.postfit_down)
    ) / 2

    # get indices to sort by decreasing impact
    sorted_indices = np.argsort(avg_postfit_impact)[::-1]
    bestfit = ranking_results.bestfit[sorted_indices]
    uncertainty = ranking_results.uncertainty[sorted_indices]
    labels = np.asarray(ranking_results.labels)[sorted_indices]  # labels are list
    prefit_up = ranking_results.prefit_up[sorted_indices]
    prefit_down = ranking_results.prefit_down[sorted_indices]
    postfit_up = ranking_results.postfit_up[sorted_indices]
    postfit_down = ranking_results.postfit_down[sorted_indices]

    if max_pars is not None:
        # only keep leading parameters in ranking
        bestfit = bestfit[:max_pars]
        uncertainty = uncertainty[:max_pars]
        labels = labels[:max_pars]
        prefit_up = prefit_up[:max_pars]
        prefit_down = prefit_down[:max_pars]
        postfit_up = postfit_up[:max_pars]
        postfit_down = postfit_down[:max_pars]

    if method == "matplotlib":
        from .contrib import matplotlib_visualize

        matplotlib_visualize.ranking(
            bestfit,
            uncertainty,
            labels,
            prefit_up,
            prefit_down,
            postfit_up,
            postfit_down,
            figure_path,
        )
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def templates(
    config: Dict[str, Any],
    figure_folder: Union[str, pathlib.Path] = "figures",
    method: str = "matplotlib",
) -> None:
    """Visualizes template histograms for systematic variations.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

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
                    {"Name": "Nominal"},
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


def scan(
    scan_results: fit.ScanResults,
    figure_folder: Union[str, pathlib.Path] = "figures",
    method: str = "matplotlib",
) -> None:
    """Visualizes the results of a likelihood scan.

    Args:
        scan_results (fit.ScanResults): results of a likelihood scan
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    # replace [], needed for staterrors
    figure_name = (
        "scan_" + scan_results.name.replace("[", "_").replace("]", "") + ".pdf"
    )
    figure_path = pathlib.Path(figure_folder) / figure_name

    if method == "matplotlib":
        from .contrib import matplotlib_visualize

        matplotlib_visualize.scan(
            scan_results.name,
            scan_results.bestfit,
            scan_results.uncertainty,
            scan_results.parameter_values,
            scan_results.delta_nlls,
            figure_path,
        )
    else:
        raise NotImplementedError(f"unknown backend: {method}")


def limit(
    limit_results: fit.LimitResults,
    figure_folder: Union[str, pathlib.Path] = "figures",
    method: str = "matplotlib",
) -> None:
    """Visualizes observed and expected CLs values as a function of the POI.

    Args:
        limit_results (fit.LimitResults): results of upper limit determination
        figure_folder (Union[str, pathlib.Path], optional): path to the folder to save
            figures in, defaults to "figures"
        method (str, optional): backend to use for plotting, defaults to "matplotlib"

    Raises:
        NotImplementedError: when trying to plot with a method that is not supported
    """
    figure_path = pathlib.Path(figure_folder) / "limit.pdf"

    if method == "matplotlib":
        from .contrib import matplotlib_visualize

        matplotlib_visualize.limit(
            limit_results.observed_CLs,
            limit_results.expected_CLs,
            limit_results.poi_values,
            figure_path,
        )
    else:
        raise NotImplementedError(f"unknown backend: {method}")
