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
