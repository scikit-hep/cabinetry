import logging
from typing import Any, KeysView

import click

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
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--method", default="uproot", help="backend for histogram production")
def templates(config_path: str, method: str) -> None:
    """Produces template histograms.

    CONFIG_PATH: path to cabinetry configuration file
    """
    _set_logging()
    cabinetry_config = cabinetry_configuration.load(config_path)
    cabinetry_template_builder.create_histograms(cabinetry_config, method=method)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def postprocess(config_path: str) -> None:
    """Post-processes template histograms.

    CONFIG_PATH: path to cabinetry configuration file
    """
    _set_logging()
    cabinetry_config = cabinetry_configuration.load(config_path)
    cabinetry_template_postprocessor.run(cabinetry_config)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("ws_path", type=click.Path(exists=False))
def workspace(config_path: str, ws_path: str) -> None:
    """Produces a ``pyhf`` workspace.

    CONFIG_PATH: path to cabinetry configuration file

    WS_PATH: where to save the workspace containing the fit model
    """
    _set_logging()
    cabinetry_config = cabinetry_configuration.load(config_path)
    ws = cabinetry_workspace.build(cabinetry_config)
    cabinetry_workspace.save(ws, ws_path)


@click.command()
@click.argument("ws_path", type=click.Path(exists=True))
@click.option("--pulls", is_flag=True, help="produce pull plot")
@click.option("--corrmat", is_flag=True, help="produce correlation matrix")
@click.option("--figfolder", default="figures/", help="folder to save figures to")
def fit(ws_path: str, pulls: bool, corrmat: bool, figfolder: str) -> None:
    """Fits a workspace and optionally visualize the results.

    WS_PATH: path to workspace used in fit
    """
    _set_logging()
    ws = cabinetry_workspace.load(ws_path)
    fit_results = cabinetry_fit.fit(ws)
    if pulls:
        cabinetry_visualize.pulls(fit_results, figfolder)
    if corrmat:
        cabinetry_visualize.correlation_matrix(fit_results, figfolder)


@click.command()
@click.argument("ws_path", type=click.Path(exists=True))
@click.option("--asimov", is_flag=True, help="fit Asimov dataset")
@click.option("--max_pars", default=10, help="maximum amount of parameters in plot")
@click.option("--figfolder", default="figures/", help="folder to save figures to")
def ranking(ws_path: str, asimov: bool, max_pars: int, figfolder: str) -> None:
    """Ranks nuisance parameters and visualizes the result.

    WS_PATH: path to workspace used in fit
    """
    _set_logging()
    ws = cabinetry_workspace.load(ws_path)
    fit_results = cabinetry_fit.fit(ws, asimov=asimov, custom=True)
    ranking_results = cabinetry_fit.ranking(ws, fit_results, asimov=asimov)
    cabinetry_visualize.ranking(ranking_results, figfolder, max_pars=max_pars)


cabinetry.add_command(templates)
cabinetry.add_command(postprocess)
cabinetry.add_command(workspace)
cabinetry.add_command(fit)
cabinetry.add_command(ranking)
