import json
import pathlib
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
@mock.patch("cabinetry.configuration.validate", autospec=True)
def test_templates(mock_validate, mock_create_histograms, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    runner = CliRunner()

    # default method
    result = runner.invoke(cli.templates, [config_path])
    assert result.exit_code == 0
    assert mock_validate.call_args_list == [((config,), {})]
    assert mock_create_histograms.call_args_list == [((config,), {"method": "uproot"})]

    # different method
    result = runner.invoke(cli.templates, ["--method", "unknown", config_path])
    assert result.exit_code == 0
    assert mock_create_histograms.call_args_list[-1] == (
        (config,),
        {"method": "unknown"},
    )


@mock.patch("cabinetry.template_postprocessor.run", autospec=True)
@mock.patch("cabinetry.configuration.validate", autospec=True)
def test_postprocess(mock_validate, mock_postprocess, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    runner = CliRunner()

    result = runner.invoke(cli.postprocess, [config_path])
    assert result.exit_code == 0
    assert mock_validate.call_args_list == [((config,), {})]
    assert mock_postprocess.call_args_list == [((config,), {})]


@mock.patch(
    "cabinetry.workspace.build", return_value={"workspace": "mock"}, autospec=True
)
@mock.patch("cabinetry.configuration.validate", autospec=True)
def test_workspace(mock_validate, mock_build, cli_helpers, tmp_path):
    config = {"General": {"Measurement": "test_config"}}

    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)

    workspace_path = str(tmp_path / "workspace.json")

    runner = CliRunner()

    result = runner.invoke(cli.workspace, [config_path, workspace_path])
    assert result.exit_code == 0
    assert mock_validate.call_args_list == [((config,), {})]
    assert mock_build.call_args_list == [((config,), {})]
    assert json.loads(pathlib.Path(workspace_path).read_text()) == {"workspace": "mock"}


@mock.patch("cabinetry.visualize.correlation_matrix", autospec=True)
@mock.patch("cabinetry.visualize.pulls", autospec=True)
@mock.patch(
    "cabinetry.fit.fit",
    return_value=fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["label"], np.asarray([[1.0]]), 1.0
    ),
    autospec=True,
)
def test_fit(mock_fit, mock_pulls, mock_corrmat, tmp_path):
    workspace = {"workspace": "mock"}
    bestfit = np.asarray([1.0])
    uncertainty = np.asarray([0.1])
    labels = ["label"]
    corr_mat = np.asarray([[1.0]])
    fit_results = fit.FitResults(bestfit, uncertainty, labels, corr_mat, 1.0)

    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    runner = CliRunner()

    # default
    result = runner.invoke(cli.fit, [workspace_path])
    assert result.exit_code == 0
    assert mock_fit.call_args_list == [
        ((workspace,), {"asimov": False, "minos": None, "goodness_of_fit": False})
    ]

    # Asimov
    result = runner.invoke(cli.fit, ["--asimov", workspace_path])
    assert result.exit_code == 0
    assert mock_fit.call_args_list[-1] == (
        (workspace,),
        {"asimov": True, "minos": None, "goodness_of_fit": False},
    )

    # MINOS for one parameter
    result = runner.invoke(cli.fit, ["--minos", "par", workspace_path])
    assert result.exit_code == 0
    assert mock_fit.call_args_list[-1] == (
        (workspace,),
        {"asimov": False, "minos": ["par"], "goodness_of_fit": False},
    )

    # MINOS for multiple parameters
    result = runner.invoke(
        cli.fit, ["--minos", "par_a", "--minos", "par_b", workspace_path]
    )
    assert result.exit_code == 0
    assert mock_fit.call_args_list[-1] == (
        (workspace,),
        {"asimov": False, "minos": ["par_a", "par_b"], "goodness_of_fit": False},
    )

    # goodness-of-fit
    result = runner.invoke(cli.fit, ["--goodness_of_fit", workspace_path])
    assert result.exit_code == 0
    assert mock_fit.call_args_list[-1] == (
        (workspace,),
        {"asimov": False, "minos": None, "goodness_of_fit": True},
    )

    # pull plot
    result = runner.invoke(cli.fit, ["--pulls", workspace_path])
    assert result.exit_code == 0
    assert mock_pulls.call_args_list == [((fit_results,), {"figure_folder": "figures"})]

    # correlation matrix plot
    result = runner.invoke(cli.fit, ["--corrmat", workspace_path])
    assert result.exit_code == 0
    assert mock_corrmat.call_args_list == [
        ((fit_results,), {"figure_folder": "figures"})
    ]

    # both plots, different folder
    result = runner.invoke(
        cli.fit, ["--figfolder", "folder", "--pulls", "--corrmat", workspace_path]
    )
    assert result.exit_code == 0
    assert mock_corrmat.call_args_list[-1] == (
        (fit_results,),
        {"figure_folder": "folder"},
    )
    assert mock_pulls.call_args_list[-1] == (
        (fit_results,),
        {"figure_folder": "folder"},
    )


@mock.patch("cabinetry.visualize.ranking", autospec=True)
@mock.patch(
    "cabinetry.fit.ranking",
    return_value=fit.RankingResults(
        np.asarray([1.0]),
        np.asarray([0.1]),
        ["label"],
        np.asarray([[1.2]]),
        np.asarray([[0.8]]),
        np.asarray([[1.1]]),
        np.asarray([[0.9]]),
    ),
    autospec=True,
)
@mock.patch(
    "cabinetry.fit.fit",
    return_value=fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["label"], np.asarray([[1.0]]), 1.0
    ),
    autospec=True,
)
def test_ranking(mock_fit, mock_rank, mock_vis, tmp_path):
    workspace = {"workspace": "mock"}
    bestfit = np.asarray([1.0])
    uncertainty = np.asarray([0.1])
    labels = ["label"]
    corr_mat = np.asarray([[1.0]])
    fit_results = fit.FitResults(bestfit, uncertainty, labels, corr_mat, 1.0)

    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    runner = CliRunner()

    # default
    result = runner.invoke(cli.ranking, [workspace_path])
    assert result.exit_code == 0
    assert mock_fit.call_args_list == [((workspace,), {"asimov": False})]
    assert mock_rank.call_args_list == [((workspace, fit_results), {"asimov": False})]
    assert mock_vis.call_count == 1
    assert np.allclose(mock_vis.call_args[0][0].prefit_up, [[1.2]])
    assert np.allclose(mock_vis.call_args[0][0].prefit_down, [[0.8]])
    assert np.allclose(mock_vis.call_args[0][0].postfit_up, [[1.1]])
    assert np.allclose(mock_vis.call_args[0][0].postfit_down, [[0.9]])
    assert mock_vis.call_args[1] == {"figure_folder": "figures", "max_pars": 10}

    # Asimov, maximum amount of parameters, custom folder
    result = runner.invoke(
        cli.ranking,
        ["--asimov", "--max_pars", 3, "--figfolder", "folder", workspace_path],
    )
    assert result.exit_code == 0
    assert mock_fit.call_args_list[-1] == ((workspace,), {"asimov": True})
    assert mock_rank.call_args_list[-1] == ((workspace, fit_results), {"asimov": True})
    assert mock_vis.call_args_list[-1][1] == {"figure_folder": "folder", "max_pars": 3}


@mock.patch("cabinetry.visualize.scan", autospec=True)
@mock.patch(
    "cabinetry.fit.scan",
    return_value=fit.ScanResults("par", 1.0, 0.1, np.asarray([1.5]), np.asarray([3.5])),
    autospec=True,
)
def test_scan(mock_scan, mock_vis, tmp_path):
    workspace = {"workspace": "mock"}
    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    par_name = "par"
    scan_results = fit.ScanResults(
        par_name, 1.0, 0.1, np.asarray([1.5]), np.asarray([3.5])
    )

    runner = CliRunner()

    # default
    result = runner.invoke(cli.scan, [workspace_path, par_name])
    assert result.exit_code == 0
    assert mock_scan.call_args_list == [
        ((workspace, par_name), {"par_range": None, "n_steps": 11, "asimov": False})
    ]
    assert mock_vis.call_count == 1
    assert mock_vis.call_args[0][0].name == scan_results.name
    assert mock_vis.call_args[0][0].bestfit == scan_results.bestfit
    assert mock_vis.call_args[0][0].uncertainty == scan_results.uncertainty
    assert np.allclose(
        mock_vis.call_args[0][0].parameter_values, scan_results.parameter_values
    )
    assert np.allclose(mock_vis.call_args[0][0].delta_nlls, scan_results.delta_nlls)
    assert mock_vis.call_args[1] == {"figure_folder": "figures"}

    # only one bound
    with pytest.raises(
        ValueError,
        match="Need to either specify both lower_bound and upper_bound, or neither.",
    ):
        runner.invoke(
            cli.scan,
            ["--lower_bound", 1.0, workspace_path, par_name],
            catch_exceptions=False,
        )
    with pytest.raises(
        ValueError,
        match="Need to either specify both lower_bound and upper_bound, or neither.",
    ):
        runner.invoke(
            cli.scan,
            ["--upper_bound", 1.0, workspace_path, par_name],
            catch_exceptions=False,
        )

    # custom bounds, number of steps and Asimov
    result = runner.invoke(
        cli.scan,
        [
            "--lower_bound",
            0.0,
            "--upper_bound",
            2.0,
            "--n_steps",
            21,
            "--asimov",
            "--figfolder",
            "folder",
            workspace_path,
            par_name,
        ],
    )
    assert result.exit_code == 0
    assert mock_scan.call_args_list[-1] == (
        (workspace, par_name),
        {"par_range": (0.0, 2.0), "n_steps": 21, "asimov": True},
    )
    assert mock_vis.call_args[1] == {"figure_folder": "folder"}


@mock.patch("cabinetry.visualize.limit", autospec=True)
@mock.patch(
    "cabinetry.fit.limit",
    return_value=fit.LimitResults(
        3.0,
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.asarray([0.05]),
        np.asarray([0.01, 0.02, 0.05, 0.07, 0.10]),
        np.asarray([3.0]),
    ),
    autospec=True,
)
def test_limit(mock_limit, mock_vis, tmp_path):
    workspace = {"workspace": "mock"}
    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    limit_results = fit.LimitResults(
        3.0,
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.asarray([0.05]),
        np.asarray([0.01, 0.02, 0.05, 0.07, 0.10]),
        np.asarray([3.0]),
    )

    runner = CliRunner()

    # default
    result = runner.invoke(cli.limit, [workspace_path])
    assert result.exit_code == 0
    assert mock_limit.call_args_list == [
        ((workspace,), {"asimov": False, "tolerance": 0.01})
    ]
    assert mock_vis.call_count == 1
    assert np.allclose(
        mock_vis.call_args[0][0].observed_limit, limit_results.observed_limit
    )
    assert np.allclose(
        mock_vis.call_args[0][0].expected_limit, limit_results.expected_limit
    )
    assert np.allclose(
        mock_vis.call_args[0][0].observed_CLs, limit_results.observed_CLs
    )
    assert np.allclose(
        mock_vis.call_args[0][0].expected_CLs, limit_results.expected_CLs
    )
    assert np.allclose(mock_vis.call_args[0][0].poi_values, limit_results.poi_values)
    assert mock_vis.call_args[1] == {"figure_folder": "figures"}

    # Asimov, tolerance, custom folder
    result = runner.invoke(
        cli.limit,
        ["--asimov", "--tolerance", "0.1", "--figfolder", "folder", workspace_path],
    )
    assert result.exit_code == 0
    assert mock_limit.call_args_list[-1] == (
        (workspace,),
        {"asimov": True, "tolerance": 0.1},
    )
    assert mock_vis.call_args_list[-1][1] == {"figure_folder": "folder"}
