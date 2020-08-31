from unittest import mock

from click.testing import CliRunner
import numpy as np
import pytest
import yaml

from cabinetry import cli
from cabinetry import fit


class CLIHelpers:
    @staticmethod
    def write_config(path, config):
        with open(path, "w") as f:
            yaml.dump(config, f)


@pytest.fixture
def cli_helpers():
    return CLIHelpers


def test_cabinetry():
    runner = CliRunner()
    result = runner.invoke(cli.cabinetry, ["--help"])
    assert result.exit_code == 0
    assert "Entrypoint to the cabinetry CLI." in result.output


# using autospec to catch changes in public API
@mock.patch("cabinetry.template_builder.create_histograms", autospec=True)
@mock.patch(
    "cabinetry.configuration.read",
    return_value={"General": {"Measurement": "test_config"}},
    autospec=True,
)
def test_templates(mock_read, mock_create_histograms, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    runner = CliRunner()

    # default method
    result = runner.invoke(cli.templates, [config_path])
    assert result.exit_code == 0
    assert mock_read.call_args_list == [((config_path,), {})]
    assert mock_create_histograms.call_args_list == [((config,), {"method": "uproot"})]

    # different method
    result = runner.invoke(cli.templates, ["--method", "unknown", config_path])
    assert result.exit_code == 0
    assert mock_create_histograms.call_args_list[-1] == (
        (config,),
        {"method": "unknown"},
    )


@mock.patch("cabinetry.template_postprocessor.run", autospec=True)
@mock.patch(
    "cabinetry.configuration.read",
    return_value={"General": {"Measurement": "test_config"}},
    autospec=True,
)
def test_postprocess(mock_read, mock_postprocess, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    runner = CliRunner()

    result = runner.invoke(cli.postprocess, [config_path])
    assert result.exit_code == 0
    assert mock_read.call_args_list == [((config_path,), {})]
    assert mock_postprocess.call_args_list == [((config,), {})]


@mock.patch("cabinetry.workspace.save", autospec=True)
@mock.patch(
    "cabinetry.workspace.build", return_value={"workspace": "mock"}, autospec=True
)
@mock.patch(
    "cabinetry.configuration.read",
    return_value={"General": {"Measurement": "test_config"}},
    autospec=True,
)
def test_workspace(mock_read, mock_build, mock_save, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    workspace_path = str(tmp_path / "workspace.json")

    runner = CliRunner()

    # default histogram folder
    result = runner.invoke(cli.workspace, [config_path, workspace_path])
    assert result.exit_code == 0
    assert mock_read.call_args_list == [((config_path,), {})]
    assert mock_build.call_args_list == [((config,), {})]
    assert mock_save.call_args_list == [(({"workspace": "mock"}, workspace_path), {})]


@mock.patch("cabinetry.visualize.correlation_matrix", autospec=True)
@mock.patch("cabinetry.visualize.pulls", autospec=True)
@mock.patch(
    "cabinetry.fit.fit",
    return_value=fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["label"], np.asarray([[1.0]]), 1.0
    ),
    autospec=True,
)
@mock.patch(
    "cabinetry.workspace.load", return_value={"workspace": "mock"}, autospec=True
)
def test_fit(mock_load, mock_fit, mock_pulls, mock_corrmat, tmp_path):
    workspace = {"workspace": "mock"}
    bestfit = np.asarray([1.0])
    uncertainty = np.asarray([0.1])
    labels = ["label"]
    corr_mat = np.asarray([[1.0]])
    fit_results = fit.FitResults(bestfit, uncertainty, labels, corr_mat, 1.0)

    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write("{'workspace': 'mock'}")

    runner = CliRunner()

    # default
    result = runner.invoke(cli.fit, [workspace_path])
    assert result.exit_code == 0
    assert mock_load.call_args_list == [((workspace_path,), {})]
    assert mock_fit.call_args_list == [((workspace,), {})]

    # pull plot
    result = runner.invoke(cli.fit, ["--pulls", workspace_path])
    assert result.exit_code == 0
    assert mock_pulls.call_args_list == [((fit_results, "figures/"), {})]

    # correlation matrix plot
    result = runner.invoke(cli.fit, ["--corrmat", workspace_path])
    assert result.exit_code == 0
    assert mock_corrmat.call_args_list == [((fit_results, "figures/"), {})]

    # both plots, different folder
    result = runner.invoke(
        cli.fit, ["--figfolder", "folder/", "--pulls", "--corrmat", workspace_path]
    )
    assert result.exit_code == 0
    assert mock_corrmat.call_args_list[-1] == ((fit_results, "folder/"), {})
    assert mock_pulls.call_args_list[-1] == ((fit_results, "folder/"), {})
