"""Implements the command line interface."""

import io
import json
import logging
import pathlib
from typing import Any, List, Optional, Tuple

import click
import yaml

from cabinetry import __version__
from cabinetry import configuration as cabinetry_configuration
from cabinetry import fit as cabinetry_fit
from cabinetry import model_utils as cabinetry_model_utils
from cabinetry import templates as cabinetry_templates
from cabinetry import visualize as cabinetry_visualize
from cabinetry import workspace as cabinetry_workspace


class OrderedGroup(click.Group):
    """A group that shows commands in the order they were added."""

    def list_commands(self, _: Any) -> List[str]:
        """Returns a list of commands."""
        return list(self.commands.keys())


def _set_logging() -> None:
    """Sets log levels and format for CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
    )
    logging.getLogger("pyhf").setLevel(logging.WARNING)


@click.version_option(version=__version__)
@click.group(cls=OrderedGroup)
def cabinetry() -> None:
    """Entrypoint to the cabinetry CLI."""


@click.command()
@click.argument("config", type=click.File("r"))
@click.option(
    "--method",
    default="uproot",
    help="backend for histogram production (default: uproot)",
)
def templates(config: io.TextIOWrapper, method: str) -> None:
    """Produces template histograms.

    CONFIG: path to cabinetry configuration file
    """
    _set_logging()
    cabinetry_config = yaml.safe_load(config)
    cabinetry_configuration.validate(cabinetry_config)
    cabinetry_templates.build(cabinetry_config, method=method)


@click.command()
@click.argument("config", type=click.File("r"))
def postprocess(config: io.TextIOWrapper) -> None:
    """Post-processes template histograms.

    CONFIG: path to cabinetry configuration file
    """
    _set_logging()
    cabinetry_config = yaml.safe_load(config)
    cabinetry_configuration.validate(cabinetry_config)
    cabinetry_templates.postprocess(cabinetry_config)


@click.command()
@click.argument("config", type=click.File("r"))
@click.argument("ws_spec", type=click.File("w"))
def workspace(config: io.TextIOWrapper, ws_spec: io.TextIOWrapper) -> None:
    """Produces a ``pyhf`` workspace.

    CONFIG: path to cabinetry configuration file

    WS_SPEC: where to save the workspace containing the fit model
    """
    _set_logging()
    cabinetry_config = yaml.safe_load(config)
    cabinetry_configuration.validate(cabinetry_config)
    ws = cabinetry_workspace.build(cabinetry_config)
    # create folder containing workspace if needed
    pathlib.Path(ws_spec.name).parent.mkdir(parents=True, exist_ok=True)
    ws_spec.write(json.dumps(ws, sort_keys=True, indent=4))


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
@click.option(
    "--minos",
    type=str,
    multiple=True,
    help="run MINOS for a parameter (default: disabled)",
)
@click.option(
    "--goodness_of_fit", is_flag=True, help="calculate goodness-of-fit (default: False)"
)
@click.option("--pulls", is_flag=True, help="produce pull plot (default: False)")
@click.option(
    "--corrmat", is_flag=True, help="produce correlation matrix (default: False)"
)
@click.option(
    "--figfolder",
    default="figures",
    help='folder to save figures to (default: "figures")',
)
def fit(
    ws_spec: io.TextIOWrapper,
    asimov: bool,
    minos: Tuple[str, ...],
    goodness_of_fit: bool,
    pulls: bool,
    corrmat: bool,
    figfolder: str,
) -> None:
    """Fits a workspace and optionally visualizes the results.

    WS_SPEC: path to workspace used in fit
    """
    _set_logging()
    # convert minos argument to None if no parameter is specified, otherwise to a list
    if len(minos) == 0:
        minos_converted = None
    else:
        minos_converted = list(minos)
    ws = json.load(ws_spec)
    model, data = cabinetry_model_utils.model_and_data(ws, asimov=asimov)
    fit_results = cabinetry_fit.fit(
        model, data, minos=minos_converted, goodness_of_fit=goodness_of_fit
    )
    if pulls:
        cabinetry_visualize.pulls(fit_results, figure_folder=figfolder)
    if corrmat:
        cabinetry_visualize.correlation_matrix(fit_results, figure_folder=figfolder)
    pass  # fixes coverage, see https://github.com/nedbat/coveragepy/issues/198


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
@click.option(
    "--max_pars", default=10, help="maximum amount of parameters in plot (default: 10)"
)
@click.option(
    "--figfolder",
    default="figures",
    help='folder to save figures to (default: "figures")',
)
def ranking(
    ws_spec: io.TextIOWrapper, asimov: bool, max_pars: int, figfolder: str
) -> None:
    """Ranks nuisance parameters and visualizes the result.

    WS_SPEC: path to workspace used in fit
    """
    _set_logging()
    ws = json.load(ws_spec)
    model, data = cabinetry_model_utils.model_and_data(ws, asimov=asimov)
    fit_results = cabinetry_fit.fit(model, data)
    ranking_results = cabinetry_fit.ranking(model, data, fit_results=fit_results)
    cabinetry_visualize.ranking(
        ranking_results, figure_folder=figfolder, max_pars=max_pars
    )


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.argument("par_name", type=str)
@click.option(
    "--lower_bound",
    default=None,
    type=float,
    help="lower parameter bound in scan (default: auto)",
)
@click.option(
    "--upper_bound",
    default=None,
    type=float,
    help="upper parameter bound in scan (default: auto)",
)
@click.option("--n_steps", default=11, help="number of steps in scan (default: 11)")
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
@click.option(
    "--figfolder",
    default="figures",
    help='folder to save figures to (default: "figures")',
)
def scan(
    ws_spec: io.TextIOWrapper,
    par_name: str,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    n_steps: int,
    asimov: bool,
    figfolder: str,
) -> None:
    """Performs and visualizes a likelihood scan over a parameter.

    Parameter bounds are determined automatically, unless both the ``lower_bound`` and
    ``upper_bound`` parameters are provided.

    WS_SPEC: path to workspace used in fit

    PAR_NAME: name of parameter to scan over
    """
    _set_logging()
    par_range: Optional[Tuple[float, float]]
    if (lower_bound is not None) and (upper_bound is not None):
        # both bounds specified
        par_range = (lower_bound, upper_bound)
    elif (lower_bound is not None) or (upper_bound is not None):
        # mixed case not supported
        raise ValueError(
            "Need to either specify both lower_bound and upper_bound, or neither."
        )
    else:
        # no bounds specified
        par_range = None

    ws = json.load(ws_spec)
    model, data = cabinetry_model_utils.model_and_data(ws, asimov=asimov)
    scan_results = cabinetry_fit.scan(
        model, data, par_name, par_range=par_range, n_steps=n_steps
    )
    cabinetry_visualize.scan(scan_results, figure_folder=figfolder)


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
@click.option(
    "--tolerance",
    default=0.01,
    help="tolerance for convergence to CLs=1-confidence_level (default: 0.01)",
)
@click.option(
    "--confidence_level",
    "--cl",
    default=0.95,
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    help="confidence level for parameter limits (default: 0.95)",
)
@click.option(
    "--figfolder",
    default="figures",
    help='folder to save figures to (default: "figures")',
)
def limit(
    ws_spec: io.TextIOWrapper,
    asimov: bool,
    tolerance: float,
    confidence_level: float,
    figfolder: str,
) -> None:
    """Calculates upper limits and visualizes CLs distribution.

    WS_SPEC: path to workspace used in fit
    """
    _set_logging()
    ws = json.load(ws_spec)
    model, data = cabinetry_model_utils.model_and_data(ws, asimov=asimov)
    limit_results = cabinetry_fit.limit(
        model, data, tolerance=tolerance, confidence_level=confidence_level
    )
    cabinetry_visualize.limit(limit_results, figure_folder=figfolder)


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
def significance(ws_spec: io.TextIOWrapper, asimov: bool) -> None:
    """Calculates observed and expected discovery significance.

    WS_SPEC: path to workspace used in fit
    """
    _set_logging()
    ws = json.load(ws_spec)
    model, data = cabinetry_model_utils.model_and_data(ws, asimov=asimov)
    _ = cabinetry_fit.significance(model, data)


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--config", type=click.File("r"), help="cabinetry configuration file")
@click.option(
    "--postfit", is_flag=True, help="visualize post-fit model (default: pre-fit model)"
)
@click.option(
    "--figfolder",
    default="figures",
    help='folder to save figures to (default: "figures")',
)
def data_mc(
    ws_spec: io.TextIOWrapper,
    config: Optional[io.TextIOWrapper],
    postfit: bool,
    figfolder: str,
) -> None:
    """Visualizes distributions of fit model and observed data.

    WS_SPEC: path to workspace
    """
    _set_logging()
    ws = json.load(ws_spec)
    model, data = cabinetry_model_utils.model_and_data(ws)

    if config is not None:
        cabinetry_config = yaml.safe_load(config)
        cabinetry_configuration.validate(cabinetry_config)
    else:
        cabinetry_config = None

    # optionally perform maximum likelihood fit to obtain post-fit model
    fit_results = cabinetry_fit.fit(model, data) if postfit else None

    model_prediction = cabinetry_model_utils.prediction(model, fit_results=fit_results)
    cabinetry_visualize.data_mc(
        model_prediction,
        data,
        config=cabinetry_config,
        figure_folder=figfolder,
        close_figure=True,
        save_figure=True,
    )


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option(
    "--split_by_sample",
    is_flag=True,
    help="split grids by sample (default: split by channel)",
)
@click.option(
    "--figfolder",
    default="figures",
    help='folder to save figures to (default: "figures")',
)
def modifier_grid(
    ws_spec: io.TextIOWrapper,
    split_by_sample: bool,
    figfolder: str,
) -> None:
    """Visualizes modifier structure of a model.

    WS_SPEC: path to workspace
    """
    _set_logging()
    ws = json.load(ws_spec)
    model, _ = cabinetry_model_utils.model_and_data(ws)
    cabinetry_visualize.modifier_grid(
        model,
        figure_folder=figfolder,
        split_by_sample=split_by_sample,
        close_figure=True,
        save_figure=True,
    )


cabinetry.add_command(templates)
cabinetry.add_command(postprocess)
cabinetry.add_command(workspace)
cabinetry.add_command(fit)
cabinetry.add_command(ranking)
cabinetry.add_command(scan)
cabinetry.add_command(limit)
cabinetry.add_command(significance)
cabinetry.add_command(data_mc)
cabinetry.add_command(modifier_grid)
