import pathlib
from unittest import mock

import pytest

from cabinetry import histo
from cabinetry import workspace


def test_WorkspaceBuilder():
    config = {"General": {"HistogramFolder": "path/"}}
    ws_builder = workspace.WorkspaceBuilder(config)
    assert ws_builder.config == config
    assert ws_builder.histogram_folder == pathlib.Path("path/")


def test_WorkspaceBuilder__get_data_sample():
    mc_sample = {"Name": "MC"}
    data_sample = {"Name": "Data", "Data": True}

    example_config = {
        "General": {"HistogramFolder": "path"},
        "Samples": [mc_sample, data_sample],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder._get_data_sample() == data_sample

    config_two_data_samples = {
        "General": {"HistogramFolder": "path"},
        "Samples": [data_sample, data_sample],
    }
    ws_builder = workspace.WorkspaceBuilder(config_two_data_samples)
    with pytest.raises(ValueError, match="did not find exactly one data sample"):
        ws_builder._get_data_sample()


def test_WorkspaceBuilder__get_constant_parameter_setting():
    config_no_fixed = {"General": {"HistogramFolder": "path"}}
    ws_builder = workspace.WorkspaceBuilder(config_no_fixed)
    assert ws_builder._get_constant_parameter_setting("par") is None

    config_others_fixed = {
        "General": {
            "HistogramFolder": "path",
            "Fixed": [{"Name": "par_a", "Value": 1.2}],
        }
    }
    ws_builder = workspace.WorkspaceBuilder(config_others_fixed)
    assert ws_builder._get_constant_parameter_setting("par_b") is None

    config_par_fixed = {
        "General": {
            "HistogramFolder": "path",
            "Fixed": [{"Name": "par_a", "Value": 1.2}],
        }
    }
    ws_builder = workspace.WorkspaceBuilder(config_par_fixed)
    assert ws_builder._get_constant_parameter_setting("par_a") == 1.2


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    return_value=histo.Histogram.from_arrays([0, 1, 2], [1.0, 2.0], [0.1, 0.1]),
)
def test_WorkspaceBuilder_get_yield_for_sample(mock_histogram):
    expected_yields = [1.0, 2.0]
    ws_builder = workspace.WorkspaceBuilder({"General": {"HistogramFolder": "path"}})
    yields = ws_builder.get_yield_for_sample({"Name": "region"}, {"Name": "signal"})
    assert yields == expected_yields

    # non-nominal
    yields_non_nominal = ws_builder.get_yield_for_sample(
        {"Name": "region"}, {"Name": "signal"}, systematic={"Name": "variation"}
    )
    assert yields_non_nominal == expected_yields

    assert mock_histogram.call_args_list == [
        (
            (
                pathlib.Path("path"),
                {"Name": "region"},
                {"Name": "signal"},
                {"Name": "Nominal"},
            ),
            {"modified": True},
        ),
        (
            (
                pathlib.Path("path"),
                {"Name": "region"},
                {"Name": "signal"},
                {"Name": "variation"},
            ),
            {"modified": True},
        ),
    ]


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    return_value=histo.Histogram.from_arrays([0, 1, 2], [1.0, 2.0], [0.1, 0.1]),
)
def test_WorkspaceBuilder_get_unc_for_sample(mock_histogram):
    expected_unc = [0.1, 0.1]
    ws_builder = workspace.WorkspaceBuilder({"General": {"HistogramFolder": "path"}})
    unc = ws_builder.get_unc_for_sample({"Name": "region"}, {"Name": "signal"})
    assert unc == expected_unc

    # non-nominal
    unc_non_nominal = ws_builder.get_unc_for_sample(
        {"Name": "region"}, {"Name": "signal"}, systematic={"Name": "variation"}
    )
    assert unc_non_nominal == expected_unc

    assert mock_histogram.call_args_list == [
        (
            (
                pathlib.Path("path"),
                {"Name": "region"},
                {"Name": "signal"},
                {"Name": "Nominal"},
            ),
            {"modified": True},
        ),
        (
            (
                pathlib.Path("path"),
                {"Name": "region"},
                {"Name": "signal"},
                {"Name": "variation"},
            ),
            {"modified": True},
        ),
    ]


def test_WorkspaceBuilder_get_NF_modifiers():
    # one NF affects sample
    example_config = {
        "General": {"HistogramFolder": "path"},
        "NormFactors": [{"Name": "mu", "Samples": ["ABC", "DEF"]}],
    }
    sample = {"Name": "DEF"}
    expected_modifier = [{"data": None, "name": "mu", "type": "normfactor"}]
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.get_NF_modifiers(sample) == expected_modifier

    # no NF affects sample
    sample = {"Name": "GHI"}
    assert ws_builder.get_NF_modifiers(sample) == []

    # multiple NFs affect sample
    example_config = {
        "General": {"HistogramFolder": "path"},
        "NormFactors": [
            {"Name": "mu", "Samples": ["ABC", "DEF"]},
            {"Name": "k", "Samples": ["DEF", "GHI"]},
        ],
    }
    sample = {"Name": "DEF"}
    expected_modifier = [
        {"data": None, "name": "mu", "type": "normfactor"},
        {"data": None, "name": "k", "type": "normfactor"},
    ]
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.get_NF_modifiers(sample) == expected_modifier


def test_WorkspaceBuilder_get_Normalization_modifier():
    systematic = {
        "Name": "sys",
        "Up": {"Normalization": 0.1},
        "Down": {"Normalization": -0.05},
    }
    expected_modifier = {
        "name": "sys",
        "type": "normsys",
        "data": {"hi": 1.1, "lo": 0.95},
    }
    ws_builder = workspace.WorkspaceBuilder({"General": {"HistogramFolder": "path"}})
    assert ws_builder.get_Normalization_modifier(systematic) == expected_modifier


def test_WorkspaceBuilder_get_NormPlusShape_modifiers():
    ...


def test_WorkspaceBuilder_get_sys_modifiers():
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Systematics": [
            {
                "Name": "sys",
                "Type": "Normalization",
                "Samples": "Signal",
                "Up": {"Normalization": 0.1},
                "Down": {"Normalization": -0.05},
            }
        ],
    }
    sample = {"Name": "Signal"}
    region = {}
    # needs to be expanded to include histogram loading
    ws_builder = workspace.WorkspaceBuilder(example_config)
    modifiers = ws_builder.get_sys_modifiers(region, sample)
    expected_modifiers = [
        {"name": "sys", "type": "normsys", "data": {"hi": 1.1, "lo": 0.95}}
    ]
    assert modifiers == expected_modifiers

    # unsupported systematics type
    example_config_unsupported = {
        "General": {"HistogramFolder": "path"},
        "Systematics": [
            {"Name": "Normalization", "Type": "unknown", "Samples": "Signal"}
        ],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config_unsupported)
    with pytest.raises(
        NotImplementedError, match="not supporting other systematic types yet"
    ):
        ws_builder.get_sys_modifiers(region, sample)

    # need an extra test for NormPlusShape
    ...


@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.get_unc_for_sample", return_value=[0.1, 0.1]
)
@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.get_yield_for_sample",
    return_value=[1.0, 2.0],
)
def test_WorkspaceBuilder_get_channels(mock_get_yield, mock_get_unc):
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Regions": [{"Name": "region_1"}],
        "Samples": [{"Name": "signal"}, {"Data": True}],
        "NormFactors": [],
    }

    ws_builder = workspace.WorkspaceBuilder(example_config)
    channels = ws_builder.get_channels()
    expected_channels = [
        {
            "name": "region_1",
            "samples": [
                {
                    "name": "signal",
                    "data": [1.0, 2.0],
                    "modifiers": [
                        {
                            "name": "staterror_region_1",
                            "type": "staterror",
                            "data": [0.1, 0.1],
                        }
                    ],
                }
            ],
        }
    ]
    assert channels == expected_channels
    assert mock_get_yield.call_args_list == [
        ((example_config["Regions"][0], example_config["Samples"][0]), {})
    ]
    assert mock_get_unc.call_args_list == [
        ((example_config["Regions"][0], example_config["Samples"][0]), {})
    ]


def test_WorkspaceBuilder_get_measurement():
    example_config = {
        "General": {"Measurement": "fit", "POI": "mu", "HistogramFolder": "path"},
        "NormFactors": [
            {"Name": "NF", "Nominal": 1.0, "Bounds": [0.0, 5.0], "Fixed": False}
        ],
    }
    expected_measurement = [
        {
            "name": "fit",
            "config": {
                "poi": "mu",
                "parameters": [{"name": "NF", "inits": [1.0], "bounds": [[0.0, 5.0]]}],
            },
        }
    ]
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.get_measurements() == expected_measurement

    # no norm factor settings
    example_config_no_NF_settings = {
        "General": {"Measurement": "fit", "POI": "mu", "HistogramFolder": "path"},
        "NormFactors": [{"Name": "NF"}],
    }
    expected_measurement_no_NF_settings = [
        {"name": "fit", "config": {"poi": "mu", "parameters": [{"name": "NF"}]}}
    ]
    ws_builder = workspace.WorkspaceBuilder(example_config_no_NF_settings)
    assert ws_builder.get_measurements() == expected_measurement_no_NF_settings

    # constant normfactor
    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder._get_constant_parameter_setting",
        return_value=1.2,
    ) as mock_find_const:
        expected_measurement_const_NF = [
            {
                "name": "fit",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "NF", "fixed": True, "inits": [1.2]}],
                },
            }
        ]
        # same config, but patched function to treat NF as fixed
        ws_builder = workspace.WorkspaceBuilder(example_config_no_NF_settings)
        assert ws_builder.get_measurements() == expected_measurement_const_NF
        assert mock_find_const.call_args_list == [(("NF",), {})]

    # constant systematic
    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder._get_constant_parameter_setting",
        return_value=1.2,
    ) as mock_find_const:
        example_config_const_sys = {
            "General": {
                "Measurement": "fit",
                "POI": "mu",
                "Fixed": [{"Name": "par_A", "Value": 1.2}],
                "HistogramFolder": "path",
            },
            "Systematics": [{"Name": "par_A"}],
        }
        expected_measurement_const_sys = [
            {
                "name": "fit",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "par_A", "fixed": True, "inits": [1.2]}],
                },
            }
        ]
        ws_builder = workspace.WorkspaceBuilder(example_config_const_sys)
        assert ws_builder.get_measurements() == expected_measurement_const_sys
        assert mock_find_const.call_args_list == [(("par_A",), {})]

    # no constant systematic
    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder._get_constant_parameter_setting",
        return_value=None,
    ) as mock_find_const:
        example_config_sys = {
            "General": {"Measurement": "fit", "POI": "mu", "HistogramFolder": "path"},
            "Systematics": [{"Name": "par_A"}],
        }
        expected_measurement_sys = [
            {"name": "fit", "config": {"poi": "mu", "parameters": []}}
        ]
        ws_builder = workspace.WorkspaceBuilder(example_config_sys)
        assert ws_builder.get_measurements() == expected_measurement_sys
        assert mock_find_const.call_args_list == [(("par_A",), {})]


@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.get_yield_for_sample",
    side_effect=[[1.0, 2.0], [5.0], [3.0]],
)
def test_WorkspaceBuilder_get_observations(mock_get_yield):
    # create observations list from config
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Samples": [{"Name": "data", "Data": True}],
        "Regions": [{"Name": "test_region"}],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config)
    obs = ws_builder.get_observations()
    expected_obs = [{"name": "test_region", "data": [1.0, 2.0]}]
    assert obs == expected_obs
    assert mock_get_yield.call_args_list == [
        ((example_config["Regions"][0], example_config["Samples"][0]), {})
    ]

    mock_get_yield.call_args_list = []  # rest call arguments list

    # multiple channels
    multi_channel_config = {
        "General": {"HistogramFolder": "path"},
        "Samples": [{"Name": "data", "Data": True}],
        "Regions": [{"Name": "test_region"}, {"Name": "other_region"}],
    }
    ws_builder = workspace.WorkspaceBuilder(multi_channel_config)
    obs = ws_builder.get_observations()
    expected_obs = [
        {"name": "test_region", "data": [5.0]},
        {"name": "other_region", "data": [3.0]},
    ]
    assert obs == expected_obs
    assert mock_get_yield.call_args_list == [
        ((multi_channel_config["Regions"][0], multi_channel_config["Samples"][0]), {}),
        ((multi_channel_config["Regions"][1], multi_channel_config["Samples"][0]), {}),
    ]


@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.get_observations",
    return_value=[{"name: observations"}],
)
@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.get_measurements",
    return_value=[{"name: measurement"}],
)
@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.get_channels",
    return_value=[{"name: channel"}],
)
def test_WorkspaceBuilder_build(mock_channels, mock_measuremets, mock_observations):
    ws_builder = workspace.WorkspaceBuilder({"General": {"HistogramFolder": "path"}})
    ws = ws_builder.build()
    ws_expected = {
        "channels": [{"name: channel"}],
        "measurements": [{"name: measurement"}],
        "observations": [{"name: observations"}],
        "version": "1.0.0",
    }
    assert ws == ws_expected


def test_build():
    minimal_config = {"General": {"Measurement": "test"}}

    mock_builder_instance = mock.MagicMock()
    mock_builder_instance.build.return_value = {"name": "workspace"}

    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder", return_value=mock_builder_instance
    ) as mock_builder:
        # without validation
        ws = workspace.build(minimal_config, with_validation=False)
        ws_expected = {"name": "workspace"}
        assert ws == ws_expected
        assert mock_builder.call_args_list == [((minimal_config,), {})]

        # including validation
        with mock.patch("cabinetry.workspace.validate") as mock_validate:
            ws = workspace.build(minimal_config, with_validation=True)
            assert ws == ws_expected
            assert mock_validate.call_args_list == [((ws,), {})]


@mock.patch("cabinetry.workspace.pyhf.Workspace")
def test_validate(mock_validate):
    test_ws = {"workspace": "test"}
    workspace.validate(test_ws)
    assert mock_validate.call_args_list == [((test_ws,), {})]


def test_save(tmp_path):
    # save to subfolder that needs to be created
    fname = tmp_path / "subdir" / "ws.json"
    ws = {"version": "1.0.0"}
    workspace.save(ws, fname)
    assert workspace.load(fname) == ws


def test_load(tmp_path):
    fname = tmp_path / "ws.json"
    ws = {"version": "1.0.0"}
    workspace.save(ws, fname)
    assert workspace.load(fname) == ws
