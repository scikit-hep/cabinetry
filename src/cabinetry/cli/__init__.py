import io
import json
import logging
from typing import Any, KeysView, Optional, Tuple

import click
import yaml

from .. import configuration as cabinetry_configuration
from .. import fit as cabinetry_fit
from .. import template_builder as cabinetry_template_builder
from .. import template_postprocessor as cabinetry_template_postprocessor
from .. import visualize as cabinetry_visualize
from .. import workspace as cabinetry_workspace


class OrderedGroup(click.Group):
    """A group that shows commands in the order they were added."""

    def list_commands(self, _: Any) -> KeysView[str]:
        return self.commands.keys()


def _set_logging() -> None:
    """Sets log levels and format for CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
    )
    logging.getLogger("pyhf").setLevel(logging.WARNING)


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

    CONFIG: (path to) cabinetry configuration file
    """
    _set_logging()
    cabinetry_config = yaml.safe_load(config)
    cabinetry_configuration.validate(cabinetry_config)
    cabinetry_template_builder.create_histograms(cabinetry_config, method=method)


@click.command()
@click.argument("config", type=click.File("r"))
def postprocess(config: io.TextIOWrapper) -> None:
    """Post-processes template histograms.

    CONFIG: (path to) cabinetry configuration file
    """
    _set_logging()
    cabinetry_config = yaml.safe_load(config)
    cabinetry_configuration.validate(cabinetry_config)
    cabinetry_template_postprocessor.run(cabinetry_config)


@click.command()
@click.argument("config", type=click.File("r"))
@click.argument("ws_spec", type=click.Path(exists=False))
def workspace(config: io.TextIOWrapper, ws_spec: str) -> None:
    """Produces a ``pyhf`` workspace.

    CONFIG: (path to) cabinetry configuration file

    WS_SPEC: where to save the workspace containing the fit model
    """
    _set_logging()
    cabinetry_config = yaml.safe_load(config)
    cabinetry_configuration.validate(cabinetry_config)
    ws = cabinetry_workspace.build(cabinetry_config)
    cabinetry_workspace.save(ws, ws_spec)


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
@click.option("--pulls", is_flag=True, help="produce pull plot (default: False)")
@click.option(
    "--corrmat", is_flag=True, help="produce correlation matrix (default: False)"
)
@click.option(
    "--figfolder",
    default="figures/",
    help="folder to save figures to (default: figures/)",
)
def fit(
    ws_spec: io.TextIOWrapper, asimov: bool, pulls: bool, corrmat: bool, figfolder: str
) -> None:
    """Fits a workspace and optionally visualize the results.

    WS_SPEC: path to workspace used in fit
    """
    _set_logging()
    ws = json.load(ws_spec)
    fit_results = cabinetry_fit.fit(ws, asimov=asimov)
    if pulls:
        cabinetry_visualize.pulls(fit_results, figfolder)
    if corrmat:
        cabinetry_visualize.correlation_matrix(fit_results, figfolder)


@click.command()
@click.argument("ws_spec", type=click.File("r"))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset (default: False)")
@click.option(
    "--max_pars", default=10, help="maximum amount of parameters in plot (default: 10)"
)
@click.option(
    "--figfolder",
    default="figures/",
    help="folder to save figures to (default: figures/)",
)
def ranking(
    ws_spec: io.TextIOWrapper, asimov: bool, max_pars: int, figfolder: str
) -> None:
    """Ranks nuisance parameters and visualizes the result.

    WS_SPEC: path to workspace used in fit
    """
    _set_logging()
    ws = json.load(ws_spec)
    fit_results = cabinetry_fit.fit(ws, asimov=asimov)
    ranking_results = cabinetry_fit.ranking(ws, fit_results, asimov=asimov)
    cabinetry_visualize.ranking(ranking_results, figfolder, max_pars=max_pars)


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
    default="figures/",
    help="folder to save figures to (default: figures/)",
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
    scan_results = cabinetry_fit.scan(
        ws, par_name, par_range=par_range, n_steps=n_steps, asimov=asimov
    )
    cabinetry_visualize.scan(scan_results, figfolder)


cabinetry.add_command(templates)
cabinetry.add_command(postprocess)
cabinetry.add_command(workspace)
cabinetry.add_command(fit)
cabinetry.add_command(ranking)
cabinetry.add_command(scan)
