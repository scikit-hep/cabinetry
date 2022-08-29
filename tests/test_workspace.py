import copy
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


def test_WorkspaceBuilder__data_sample():
    mc_sample = {"Name": "MC"}
    data_sample = {"Name": "Data", "Data": True}

    example_config = {
        "General": {"HistogramFolder": "path"},
        "Samples": [mc_sample, data_sample],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder._data_sample() == data_sample

    config_two_data_samples = {
        "General": {"HistogramFolder": "path"},
        "Samples": [data_sample, data_sample],
    }
    ws_builder = workspace.WorkspaceBuilder(config_two_data_samples)
    with pytest.raises(ValueError, match="did not find exactly one data sample"):
        ws_builder._data_sample()


def test_WorkspaceBuilder__constant_parameter_setting():
    config_no_fixed = {"General": {"HistogramFolder": "path"}}
    ws_builder = workspace.WorkspaceBuilder(config_no_fixed)
    assert ws_builder._constant_parameter_setting("par") is None

    config_others_fixed = {
        "General": {
            "HistogramFolder": "path",
            "Fixed": [{"Name": "par_a", "Value": 1.2}],
        }
    }
    ws_builder = workspace.WorkspaceBuilder(config_others_fixed)
    assert ws_builder._constant_parameter_setting("par_b") is None

    config_par_fixed = {
        "General": {
            "HistogramFolder": "path",
            "Fixed": [{"Name": "par_a", "Value": 1.2}],
        }
    }
    ws_builder = workspace.WorkspaceBuilder(config_par_fixed)
    assert ws_builder._constant_parameter_setting("par_a") == 1.2


def test_WorkspaceBuilder_normfactor_modifiers():
    # could mock region_contains_modifier / sample_contains_modifier
    # one NF affects sample
    example_config = {
        "General": {"HistogramFolder": "path"},
        "NormFactors": [{"Name": "mu", "Samples": ["ABC", "DEF"]}],
    }
    region = {"Name": "SR"}
    sample = {"Name": "DEF"}
    expected_modifier = [{"data": None, "name": "mu", "type": "normfactor"}]
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.normfactor_modifiers(region, sample) == expected_modifier

    # no NF affects sample
    sample = {"Name": "GHI"}
    assert ws_builder.normfactor_modifiers(region, sample) == []

    # NF enters in region
    example_config = {
        "General": {"HistogramFolder": "path"},
        "NormFactors": [{"Name": "mu", "Regions": "SR"}],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.normfactor_modifiers(region, sample) == expected_modifier

    # no NF due to region
    region = {"Name": "CR"}
    assert ws_builder.normfactor_modifiers(region, sample) == []

    # multiple NFs affect sample
    example_config = {
        "General": {"HistogramFolder": "path"},
        "NormFactors": [{"Name": "mu"}, {"Name": "k"}],
    }
    sample = {"Name": "DEF"}
    expected_modifier = [
        {"data": None, "name": "mu", "type": "normfactor"},
        {"data": None, "name": "k", "type": "normfactor"},
    ]
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.normfactor_modifiers(region, sample) == expected_modifier


def test_WorkspaceBuilder_normalization_modifier():
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
    assert ws_builder.normalization_modifier(systematic) == expected_modifier

    # ModifierName set
    systematic = {
        "Name": "sys",
        "Up": {"Normalization": 0.1},
        "Down": {"Normalization": -0.05},
        "ModifierName": "mod_name",
    }
    assert ws_builder.normalization_modifier(systematic)["name"] == "mod_name"


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    side_effect=[
        # without symmetrization: up, nominal, down
        histo.Histogram.from_arrays([0, 1, 2], [26.0, 24.0], [0.1, 0.1]),
        histo.Histogram.from_arrays([0, 1, 2], [20.0, 20.0], [0.1, 0.1]),
        histo.Histogram.from_arrays([0, 1, 2], [8.0, 12.0], [0.1, 0.1]),
        # for test of symmetrization: up and nominal
        histo.Histogram.from_arrays([0, 1, 2], [26.0, 24.0], [0.1, 0.1]),
        histo.Histogram.from_arrays([0, 1, 2], [20.0, 20.0], [0.1, 0.1]),
    ],
)
def test_WorkspaceBuilder_normplusshape_modifiers(mock_histogram):
    # could mock Histogram.normalize_to_yield
    # up: 26, 24 (1.25*nom)
    # nominal: 20, 20
    # down: 8, 12 (0.5*nom)
    example_config = {"General": {"HistogramFolder": "path"}}
    ws_builder = workspace.WorkspaceBuilder(example_config)
    region = {"Name": "SR"}
    sample = {"Name": "Signal"}
    systematic = {"Name": "sys", "Up": {}, "Down": {}}
    # no symmetrization
    modifiers = ws_builder.normplusshape_modifiers(region, sample, systematic)
    assert modifiers == [
        {"name": "sys", "type": "normsys", "data": {"hi": 1.25, "lo": 0.5}},
        {
            "name": "sys",
            "type": "histosys",
            "data": {"hi_data": [20.8, 19.2], "lo_data": [16.0, 24.0]},
        },
    ]
    assert mock_histogram.call_args_list == [
        (
            (pathlib.Path("path"), region, sample, systematic),
            {"template": "Up", "modified": True},
        ),
        ((pathlib.Path("path"), region, sample, {}), {"modified": True}),
        (
            (pathlib.Path("path"), region, sample, systematic),
            {"template": "Down", "modified": True},
        ),
    ]

    # down template via symmetrized up template, ModifierName set
    systematic = {
        "Name": "sys",
        "Up": {},
        "Down": {"Symmetrize": True},
        "ModifierName": "mod_name",
    }
    modifiers = ws_builder.normplusshape_modifiers(region, sample, systematic)
    assert modifiers == [
        {"name": "mod_name", "type": "normsys", "data": {"hi": 1.25, "lo": 0.75}},
        {
            "name": "mod_name",
            "type": "histosys",
            "data": {"hi_data": [20.8, 19.2], "lo_data": [19.2, 20.8]},
        },
    ]
    assert mock_histogram.call_args_list[3:] == [
        (
            (pathlib.Path("path"), region, sample, systematic),
            {"template": "Up", "modified": True},
        ),
        ((pathlib.Path("path"), region, sample, {}), {"modified": True}),
    ]


@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.normplusshape_modifiers",
    return_value=[{"mock": "norm"}, {"mock": "shape"}],
)
@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.normalization_modifier",
    return_value={"mock": "normsys"},
)
def test_WorkspaceBuilder_sys_modifiers(mock_norm, mock_norm_shape):
    # could mock region_contains_modifier / sample_contains_modifier
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Systematics": [
            {"Name": "norm", "Type": "Normalization"},
            {"Name": "norm_shape", "Type": "NormPlusShape"},
        ],
    }
    region = {"Name": "SR"}
    sample = {"Name": "Signal"}
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.sys_modifiers(region, sample) == [
        {"mock": "normsys"},
        {"mock": "norm"},
        {"mock": "shape"},
    ]
    assert mock_norm.call_args_list == [((example_config["Systematics"][0],), {})]
    assert mock_norm_shape.call_args_list == [
        ((region, sample, example_config["Systematics"][1]), {})
    ]

    # one systematic not present in region
    example_config_region_mismatch = copy.deepcopy(example_config)
    example_config_region_mismatch["Systematics"][0].update({"Regions": "CR"})
    ws_builder = workspace.WorkspaceBuilder(example_config_region_mismatch)
    assert ws_builder.sys_modifiers(region, sample) == [
        {"mock": "norm"},
        {"mock": "shape"},
    ]

    # one systematic not present in sample
    example_config_sample_mismatch = copy.deepcopy(example_config)
    example_config_sample_mismatch["Systematics"][1].update({"Samples": "Background"})
    ws_builder = workspace.WorkspaceBuilder(example_config_sample_mismatch)
    assert ws_builder.sys_modifiers(region, sample) == [{"mock": "normsys"}]

    # unsupported systematics type
    example_config_unsupported = {
        "General": {"HistogramFolder": "path"},
        "Systematics": [{"Name": "Normalization", "Type": "unknown"}],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config_unsupported)
    with pytest.raises(
        NotImplementedError, match="not supporting other systematic types yet"
    ):
        ws_builder.sys_modifiers(region, sample)


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    return_value=histo.Histogram.from_arrays([0, 1, 2], [1.0, 2.0], [0.1, 0.1]),
)
@mock.patch(
    "cabinetry.configuration.region_contains_sample", side_effect=[True, False, True]
)
def test_WorkspaceBuilder_channels(mock_contains, mock_histogram):
    # should mock normfactor_modifiers / sys_modifiers
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Regions": [{"Name": "region_1"}],
        "Samples": [{"Name": "signal"}, {"Data": True}],
        "NormFactors": [],
    }

    ws_builder = workspace.WorkspaceBuilder(example_config)
    channels = ws_builder.channels()
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
    assert mock_contains.call_args_list == [
        ((example_config["Regions"][0], example_config["Samples"][0]), {})
    ]
    assert mock_histogram.call_args_list == [
        (
            (
                pathlib.Path("path"),
                example_config["Regions"][0],
                example_config["Samples"][0],
                {},
            ),
            {"modified": True},
        )
    ]

    # run again, this time region will not contain sample due to side_effect
    channels = ws_builder.channels()
    expected_channels = [{"name": "region_1", "samples": []}]
    assert channels == expected_channels
    assert mock_contains.call_count == 2
    assert mock_contains.call_args == (
        (example_config["Regions"][0], example_config["Samples"][0]),
        {},
    )
    # no calls to read histogram content
    assert mock_histogram.call_count == 1

    # staterror creation disabled
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Regions": [{"Name": "region_1"}],
        "Samples": [{"Name": "signal", "DisableStaterror": True}],
        "NormFactors": [],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config)
    channels = ws_builder.channels()
    assert channels[0]["samples"][0]["modifiers"] == []


def test_WorkspaceBuilder_measurements():
    example_config = {
        "General": {"Measurement": "fit", "HistogramFolder": "path"},
        "NormFactors": [
            {"Name": "NF", "Nominal": 1.0, "Bounds": [0.0, 5.0], "Fixed": False}
        ],
    }
    expected_measurement = [
        {
            "name": "fit",
            "config": {
                "poi": "",
                "parameters": [{"name": "NF", "inits": [1.0], "bounds": [[0.0, 5.0]]}],
            },
        }
    ]
    ws_builder = workspace.WorkspaceBuilder(example_config)
    assert ws_builder.measurements() == expected_measurement

    # no norm factor settings, POI specified
    example_config_no_NF_settings = {
        "General": {"Measurement": "fit", "POI": "mu", "HistogramFolder": "path"},
        "NormFactors": [{"Name": "NF"}],
    }
    expected_measurement_no_NF_settings = [
        {"name": "fit", "config": {"poi": "mu", "parameters": [{"name": "NF"}]}}
    ]
    ws_builder = workspace.WorkspaceBuilder(example_config_no_NF_settings)
    assert ws_builder.measurements() == expected_measurement_no_NF_settings

    # constant normfactor
    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder._constant_parameter_setting",
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
        assert ws_builder.measurements() == expected_measurement_const_NF
        assert mock_find_const.call_args_list == [(("NF",), {})]

    # constant systematic
    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder._constant_parameter_setting",
        return_value=1.2,
    ) as mock_find_const:
        example_config_const_sys = {
            "General": {
                "Measurement": "fit",
                "Fixed": [{"Name": "par_A", "Value": 1.2}],
                "HistogramFolder": "path",
            },
            "Systematics": [{"Name": "par_A"}],
        }
        expected_measurement_const_sys = [
            {
                "name": "fit",
                "config": {
                    "poi": "",
                    "parameters": [{"name": "par_A", "fixed": True, "inits": [1.2]}],
                },
            }
        ]
        ws_builder = workspace.WorkspaceBuilder(example_config_const_sys)
        assert ws_builder.measurements() == expected_measurement_const_sys
        assert mock_find_const.call_args_list == [(("par_A",), {})]

    # no constant systematic
    with mock.patch(
        "cabinetry.workspace.WorkspaceBuilder._constant_parameter_setting",
        return_value=None,
    ) as mock_find_const:
        example_config_sys = {
            "General": {"Measurement": "fit", "HistogramFolder": "path"},
            "Systematics": [{"Name": "par_A"}],
        }
        expected_measurement_sys = [
            {"name": "fit", "config": {"poi": "", "parameters": []}}
        ]
        ws_builder = workspace.WorkspaceBuilder(example_config_sys)
        assert ws_builder.measurements() == expected_measurement_sys
        assert mock_find_const.call_args_list == [(("par_A",), {})]


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    side_effect=[
        histo.Histogram.from_arrays([0, 1, 2], [1.0, 2.0], [0.1, 0.1]),
        histo.Histogram.from_arrays([0, 1], [5.0], [0.1]),
        histo.Histogram.from_arrays([0, 1], [3.0], [0.1]),
    ],
)
def test_WorkspaceBuilder_observations(mock_histogram):
    # could mock _data_sample
    # create observations list from config
    example_config = {
        "General": {"HistogramFolder": "path"},
        "Samples": [{"Name": "data", "Data": True}],
        "Regions": [{"Name": "test_region"}],
    }
    ws_builder = workspace.WorkspaceBuilder(example_config)
    obs = ws_builder.observations()
    expected_obs = [{"name": "test_region", "data": [1.0, 2.0]}]
    assert obs == expected_obs
    assert mock_histogram.call_args_list == [
        (
            (
                pathlib.Path("path"),
                example_config["Regions"][0],
                example_config["Samples"][0],
                {},
            ),
            {"modified": True},
        )
    ]

    # multiple channels
    multi_channel_config = {
        "General": {"HistogramFolder": "path"},
        "Samples": [{"Name": "data", "Data": True}],
        "Regions": [{"Name": "test_region"}, {"Name": "other_region"}],
    }
    ws_builder = workspace.WorkspaceBuilder(multi_channel_config)
    obs = ws_builder.observations()
    expected_obs = [
        {"name": "test_region", "data": [5.0]},
        {"name": "other_region", "data": [3.0]},
    ]
    assert obs == expected_obs
    assert mock_histogram.call_args_list[1:] == [
        (
            (
                pathlib.Path("path"),
                multi_channel_config["Regions"][0],
                multi_channel_config["Samples"][0],
                {},
            ),
            {"modified": True},
        ),
        (
            (
                pathlib.Path("path"),
                multi_channel_config["Regions"][1],
                multi_channel_config["Samples"][0],
                {},
            ),
            {"modified": True},
        ),
    ]


@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.observations",
    return_value=[{"name: observations"}],
)
@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.measurements",
    return_value=[{"name: measurement"}],
)
@mock.patch(
    "cabinetry.workspace.WorkspaceBuilder.channels", return_value=[{"name: channel"}]
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
