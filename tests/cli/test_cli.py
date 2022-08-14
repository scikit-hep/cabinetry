import json
import pathlib
from unittest import mock

from click.testing import CliRunner
import numpy as np
import pytest
import yaml

from cabinetry import __version__
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

    result = runner.invoke(cli.cabinetry, ["--version"])
    assert result.exit_code == 0
    assert f"cabinetry, version {__version__}" in result.output


# using autospec to catch changes in public API
@mock.patch("cabinetry.templates.build", autospec=True)
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
    assert mock_create_histograms.call_args == ((config,), {"method": "unknown"})


@mock.patch("cabinetry.templates.postprocess", autospec=True)
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
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_fit(mock_utils, mock_fit, mock_pulls, mock_corrmat, tmp_path):
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
    assert mock_utils.call_args_list == [((workspace,), {"asimov": False})]
    assert mock_fit.call_args_list == [
        (("model", "data"), {"minos": None, "goodness_of_fit": False})
    ]

    # Asimov
    result = runner.invoke(cli.fit, ["--asimov", workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {"asimov": True})
    assert mock_fit.call_args == (
        ("model", "data"),
        {"minos": None, "goodness_of_fit": False},
    )

    # MINOS for one parameter
    result = runner.invoke(cli.fit, ["--minos", "par", workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {"asimov": False})
    assert mock_fit.call_args == (
        ("model", "data"),
        {"minos": ["par"], "goodness_of_fit": False},
    )

    # MINOS for multiple parameters
    result = runner.invoke(
        cli.fit, ["--minos", "par_a", "--minos", "par_b", workspace_path]
    )
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {"asimov": False})
    assert mock_fit.call_args == (
        ("model", "data"),
        {"minos": ["par_a", "par_b"], "goodness_of_fit": False},
    )

    # goodness-of-fit
    result = runner.invoke(cli.fit, ["--goodness_of_fit", workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {"asimov": False})
    assert mock_fit.call_args == (
        ("model", "data"),
        {"minos": None, "goodness_of_fit": True},
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
    assert mock_corrmat.call_args == ((fit_results,), {"figure_folder": "folder"})
    assert mock_pulls.call_args == ((fit_results,), {"figure_folder": "folder"})


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
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_ranking(mock_utils, mock_fit, mock_rank, mock_vis, tmp_path):
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
    assert mock_utils.call_args_list == [((workspace,), {"asimov": False})]
    assert mock_fit.call_args_list == [(("model", "data"), {})]
    assert mock_rank.call_args_list == [
        (("model", "data"), {"fit_results": fit_results})
    ]
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
    assert mock_utils.call_args == ((workspace,), {"asimov": True})
    assert mock_fit.call_args == (("model", "data"), {})
    assert mock_rank.call_args == (("model", "data"), {"fit_results": fit_results})
    assert mock_vis.call_args[1] == {"figure_folder": "folder", "max_pars": 3}


@mock.patch("cabinetry.visualize.scan", autospec=True)
@mock.patch(
    "cabinetry.fit.scan",
    return_value=fit.ScanResults("par", 1.0, 0.1, np.asarray([1.5]), np.asarray([3.5])),
    autospec=True,
)
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_scan(mock_utils, mock_scan, mock_vis, tmp_path):
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
    assert mock_utils.call_args_list == [((workspace,), {"asimov": False})]
    assert mock_scan.call_args_list == [
        (("model", "data", par_name), {"par_range": None, "n_steps": 11})
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
    assert mock_utils.call_args == ((workspace,), {"asimov": True})
    assert mock_scan.call_args == (
        ("model", "data", par_name),
        {"par_range": (0.0, 2.0), "n_steps": 21},
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
        0.90,
    ),
    autospec=True,
)
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_limit(mock_utils, mock_limit, mock_vis, tmp_path):
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
        0.90,
    )

    runner = CliRunner()

    # default
    result = runner.invoke(cli.limit, [workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args_list == [((workspace,), {"asimov": False})]
    assert mock_limit.call_args_list == [
        (("model", "data"), {"tolerance": 0.01, "confidence_level": 0.95})
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
    assert np.allclose(
        mock_vis.call_args[0][0].confidence_level, limit_results.confidence_level
    )
    assert mock_vis.call_args[1] == {"figure_folder": "figures"}

    # Asimov, tolerance, confidence level, custom folder
    result = runner.invoke(
        cli.limit,
        [
            "--asimov",
            "--tolerance",
            "0.1",
            "--confidence_level",
            "0.90",
            "--figfolder",
            "folder",
            workspace_path,
        ],
    )
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {"asimov": True})
    assert mock_limit.call_args == (
        ("model", "data"),
        {"tolerance": 0.1, "confidence_level": 0.90},
    )
    assert mock_vis.call_args[1] == {"figure_folder": "folder"}


@mock.patch("cabinetry.fit.significance", autospec=True)
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_significance(mock_utils, mock_sig, tmp_path):
    workspace = {"workspace": "mock"}
    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    runner = CliRunner()

    # default
    result = runner.invoke(cli.significance, [workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args_list == [((workspace,), {"asimov": False})]
    assert mock_sig.call_args_list == [(("model", "data"), {})]

    # Asimov
    result = runner.invoke(cli.significance, ["--asimov", workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {"asimov": True})
    assert mock_sig.call_args == (("model", "data"), {})


@mock.patch("cabinetry.visualize.data_mc", autospec=True)
@mock.patch(
    "cabinetry.model_utils.prediction", return_value="mock_model_pred", autospec=True
)
@mock.patch(
    "cabinetry.fit.fit",
    return_value=fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["label"], np.asarray([[1.0]]), 1.0
    ),
    autospec=True,
)
@mock.patch("cabinetry.configuration.validate", autospec=True)
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_data_mc(
    mock_utils, mock_validate, mock_fit, mock_pred, mock_vis, cli_helpers, tmp_path
):
    workspace = {"workspace": "mock"}
    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    runner = CliRunner()

    # default
    result = runner.invoke(cli.data_mc, [workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args_list == [((workspace,), {})]
    assert mock_validate.call_count == 0
    assert mock_fit.call_count == 0
    assert mock_pred.call_args_list == [(("model",), {"fit_results": None})]
    assert mock_vis.call_args_list == [
        (
            ("mock_model_pred", "data"),
            {
                "config": None,
                "figure_folder": "figures",
                "close_figure": True,
                "save_figure": True,
            },
        )
    ]

    # with config, post-fit, custom figure folder
    config = {"General": {"Measurement": "test_config"}}
    config_path = str(tmp_path / "config.yml")
    cli_helpers.write_config(config_path, config)
    fit_results = fit.FitResults(
        np.asarray([1.0]), np.asarray([0.1]), ["label"], np.asarray([[1.0]]), 1.0
    )

    result = runner.invoke(
        cli.data_mc,
        [workspace_path, "--config", config_path, "--postfit", "--figfolder", "folder"],
    )
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {})
    assert mock_validate.call_args_list == [((config,), {})]
    assert mock_fit.call_args_list == [(("model", "data"), {})]
    assert mock_pred.call_args == (("model",), {"fit_results": fit_results})
    assert mock_vis.call_args == (
        ("mock_model_pred", "data"),
        {
            "config": config,
            "figure_folder": "folder",
            "close_figure": True,
            "save_figure": True,
        },
    )


@mock.patch("cabinetry.visualize.modifier_grid", autospec=True)
@mock.patch(
    "cabinetry.model_utils.model_and_data",
    return_value=("model", "data"),
    autospec=True,
)
def test_modifier_grid(mock_utils, mock_vis, cli_helpers, tmp_path):
    workspace = {"workspace": "mock"}
    workspace_path = str(tmp_path / "workspace.json")

    # need to save workspace to file since click looks for it
    with open(workspace_path, "w") as f:
        f.write('{"workspace": "mock"}')

    runner = CliRunner()

    # default
    result = runner.invoke(cli.modifier_grid, [workspace_path])
    assert result.exit_code == 0
    assert mock_utils.call_args_list == [((workspace,), {})]
    assert mock_vis.call_args_list == [
        (
            ("model",),
            {
                "figure_folder": "figures",
                "split_by_sample": False,
                "close_figure": True,
                "save_figure": True,
            },
        )
    ]

    # split by sample, custom figure folder
    result = runner.invoke(
        cli.modifier_grid,
        [workspace_path, "--split_by_sample", "--figfolder", "folder"],
    )
    assert result.exit_code == 0
    assert mock_utils.call_args == ((workspace,), {})
    assert mock_vis.call_args == (
        ("model",),
        {
            "figure_folder": "folder",
            "split_by_sample": True,
            "close_figure": True,
            "save_figure": True,
        },
    )
