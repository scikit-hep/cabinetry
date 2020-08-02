import logging
from typing import Any, KeysView

import click

from .. import configuration as cabinetry_configuration
from .. import fit as cabinetry_fit
from .. import template_builder as cabinetry_template_builder
from .. import visualize as cabinetry_visualize
from .. import workspace as cabinetry_workspace


class OrderedGroup(click.Group):
    """a group that shows commands in the order they were added
    """

    def list_commands(self, _: Any) -> KeysView[str]:
        return self.commands.keys()


def set_logging() -> None:
    """setup log levels and format for CLI
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
    )
    logging.getLogger("pyhf").setLevel(logging.WARNING)


@click.group(cls=OrderedGroup)
def cabinetry() -> None:
    """entrypoint to the cabinetry CLI
    """
    pass


@click.command()
@click.argument("config_path")
@click.option(
    "--histofolder", default="histograms/", help="folder to save histograms to"
)
@click.option("--method", default="uproot", help="backend for histogram production")
def templates(config_path: str, histofolder: str, method: str) -> None:
    """produce template histograms
    """
    set_logging()
    config = cabinetry_configuration.read(config_path)
    cabinetry_template_builder.create_histograms(config, histofolder, method=method)


@click.command()
@click.argument("config_path")
@click.argument("ws_path")
@click.option(
    "--histofolder", default="histograms/", help="folder containing histograms"
)
def workspace(config_path: str, ws_path: str, histofolder: str) -> None:
    """produce a `pyhf` workspace
    """
    set_logging()
    config = cabinetry_configuration.read(config_path)
    ws = cabinetry_workspace.build(config, histofolder)
    cabinetry_workspace.save(ws, ws_path)


@click.command()
@click.argument("ws_path")
@click.option("--pulls", is_flag=True, help="produce pull plot")
@click.option("--corrmat", is_flag=True, help="produce correlation matrix")
@click.option("--figfolder", default="figures/", help="folder to save figures to")
def fit(ws_path: str, pulls: bool, corrmat: bool, figfolder: str) -> None:
    """fit a workspace and optionally visualize the results
    """
    set_logging()
    ws = cabinetry_workspace.load(ws_path)
    bestfit, uncertainty, labels, _, corr_mat = cabinetry_fit.fit(ws)
    if pulls:
        cabinetry_visualize.pulls(bestfit, uncertainty, labels, figfolder)
    if corrmat:
        cabinetry_visualize.correlation_matrix(corr_mat, labels, figfolder)


cabinetry.add_command(templates)
cabinetry.add_command(workspace)
cabinetry.add_command(fit)
