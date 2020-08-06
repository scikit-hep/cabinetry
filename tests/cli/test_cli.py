from unittest import mock

from click.testing import CliRunner
import pytest
import yaml

from cabinetry import cli


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


@mock.patch("cabinetry.template_builder.create_histograms")
@mock.patch(
    "cabinetry.configuration.read",
    return_value={"General": {"Measurement": "test_config"}},
)
def test_templates(mock_read, mock_create_histograms, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    runner = CliRunner()

    # default histogram folder
    result = runner.invoke(cli.templates, [config_path])
    assert result.exit_code == 0
    assert mock_read.call_args_list == [((config_path,), {})]
    assert mock_create_histograms.call_args_list == [
        ((config, "histograms/"), {"method": "uproot"})
    ]

    # specified histogram folder
    result = runner.invoke(cli.templates, ["--histofolder", "path/", config_path])
    assert result.exit_code == 0
    assert mock_create_histograms.call_args_list[-1] == (
        (config, "path/"),
        {"method": "uproot"},
    )

    # different method
    result = runner.invoke(cli.templates, ["--method", "unknown", config_path])
    assert result.exit_code == 0
    assert mock_create_histograms.call_args_list[-1] == (
        (config, "histograms/"),
        {"method": "unknown"},
    )


@mock.patch("cabinetry.template_postprocessor.run")
@mock.patch(
    "cabinetry.configuration.read",
    return_value={"General": {"Measurement": "test_config"}},
)
def test_postprocess(mock_read, mock_postprocess, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    runner = CliRunner()

    # default histogram folder
    result = runner.invoke(cli.postprocess, [config_path])
    assert result.exit_code == 0
    assert mock_read.call_args_list == [((config_path,), {})]
    assert mock_postprocess.call_args_list == [((config, "histograms/"), {})]

    # specified histogram folder
    result = runner.invoke(cli.postprocess, ["--histofolder", "path/", config_path])
    assert result.exit_code == 0
    assert mock_postprocess.call_args_list[-1] == ((config, "path/"), {})


@mock.patch("cabinetry.workspace.save")
@mock.patch("cabinetry.workspace.build", return_value={"workspace": "mock"})
@mock.patch(
    "cabinetry.configuration.read",
    return_value={"General": {"Measurement": "test_config"}},
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
    assert mock_build.call_args_list == [((config, "histograms/"), {})]
    assert mock_save.call_args_list == [(({"workspace": "mock"}, workspace_path), {})]

    # specified histogram folder
    result = runner.invoke(
        cli.workspace, ["--histofolder", "path/", config_path, workspace_path]
    )
    assert result.exit_code == 0
    assert mock_build.call_args_list[-1] == ((config, "path/"), {})


@mock.patch("cabinetry.visualize.correlation_matrix")
@mock.patch("cabinetry.visualize.pulls")
@mock.patch("cabinetry.fit.fit", return_value=([1.0], [0.1], "label", None, [[1.0]]))
@mock.patch("cabinetry.workspace.load", return_value={"workspace": "mock"})
def test_fit(mock_load, mock_fit, mock_pulls, mock_corrmat):
    workspace = {"workspace": "mock"}
    bestfit = [1.0]
    uncertainty = [0.1]
    labels = "label"
    corr_mat = [[1.0]]

    workspace_path = "workspace.json"

    runner = CliRunner()

    # default
    result = runner.invoke(cli.fit, [workspace_path])
    assert result.exit_code == 0
    assert mock_load.call_args_list == [((workspace_path,), {})]
    assert mock_fit.call_args_list == [((workspace,), {})]

    # pull plot
    result = runner.invoke(cli.fit, ["--pulls", workspace_path])
    assert result.exit_code == 0
    assert mock_pulls.call_args_list == [
        ((bestfit, uncertainty, labels, "figures/"), {})
    ]

    # correlation matrix plot
    result = runner.invoke(cli.fit, ["--corrmat", workspace_path])
    assert result.exit_code == 0
    assert mock_corrmat.call_args_list == [((corr_mat, labels, "figures/"), {})]

    # both plots, different folder
    result = runner.invoke(
        cli.fit, ["--figfolder", "folder/", "--pulls", "--corrmat", workspace_path]
    )
    assert result.exit_code == 0
    assert mock_corrmat.call_args_list[-1] == ((corr_mat, labels, "folder/"), {})
    assert mock_pulls.call_args_list[-1] == (
        (bestfit, uncertainty, labels, "folder/"),
        {},
    )
