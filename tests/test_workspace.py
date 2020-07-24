from unittest import mock

import pytest

from cabinetry import histo
from cabinetry import workspace


def test__get_data_sample():
    mc_sample = {"Name": "MC"}
    data_sample = {"Name": "Data", "Data": True}
    config_example = {"Samples": [mc_sample, data_sample]}
    assert workspace._get_data_sample(config_example) == data_sample


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    return_value=histo.Histogram.from_arrays([0, 1, 2], [1.0, 2.0], [0.1, 0.1]),
)
def test_get_yield_for_sample(mock_histogram):
    expected_yields = [1.0, 2.0]
    yields = workspace.get_yield_for_sample(
        {"Name": "region"}, {"Name": "signal"}, "path"
    )
    assert yields == expected_yields

    # non-nominal
    yields_non_nominal = workspace.get_yield_for_sample(
        {"Name": "region"}, {"Name": "signal"}, "path", systematic={"Name": "variation"}
    )
    assert yields_non_nominal == expected_yields

    assert mock_histogram.call_args_list == [
        (
            ("path", {"Name": "region"}, {"Name": "signal"}, {"Name": "nominal"}),
            {"modified": True},
        ),
        (
            ("path", {"Name": "region"}, {"Name": "signal"}, {"Name": "variation"}),
            {"modified": True},
        ),
    ]


@mock.patch(
    "cabinetry.workspace.histo.Histogram.from_config",
    return_value=histo.Histogram.from_arrays([0, 1, 2], [1.0, 2.0], [0.1, 0.1]),
)
def test_get_unc_for_sample(mock_histogram):
    expected_unc = [0.1, 0.1]
    unc = workspace.get_unc_for_sample({"Name": "region"}, {"Name": "signal"}, "path")
    assert unc == expected_unc

    # non-nominal
    unc_non_nominal = workspace.get_unc_for_sample(
        {"Name": "region"}, {"Name": "signal"}, "path", systematic={"Name": "variation"}
    )
    assert unc_non_nominal == expected_unc

    assert mock_histogram.call_args_list == [
        (
            ("path", {"Name": "region"}, {"Name": "signal"}, {"Name": "nominal"}),
            {"modified": True},
        ),
        (
            ("path", {"Name": "region"}, {"Name": "signal"}, {"Name": "variation"}),
            {"modified": True},
        ),
    ]


def test_get_NF_modifiers():
    # one NF affects sample
    example_config = {"NormFactors": [{"Name": "mu", "Samples": ["ABC", "DEF"]}]}
    sample = {"Name": "DEF"}
    expected_modifier = [{"data": None, "name": "mu", "type": "normfactor"}]
    assert workspace.get_NF_modifiers(example_config, sample) == expected_modifier

    # no NF affects sample
    sample = {"Name": "GHI"}
    assert workspace.get_NF_modifiers(example_config, sample) == []

    # multiple NFs affect sample
    example_config = {
        "NormFactors": [
            {"Name": "mu", "Samples": ["ABC", "DEF"]},
            {"Name": "k", "Samples": ["DEF", "GHI"]},
        ]
    }
    sample = {"Name": "DEF"}
    expected_modifier = [
        {"data": None, "name": "mu", "type": "normfactor"},
        {"data": None, "name": "k", "type": "normfactor"},
    ]
    assert workspace.get_NF_modifiers(example_config, sample) == expected_modifier


def test_get_Normalization_modifier():
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
    assert workspace.get_Normalization_modifier(systematic) == expected_modifier


def test_get_sys_modifiers():
    config_example = {
        "Systematics": [
            {
                "Name": "sys",
                "Type": "Normalization",
                "Samples": "Signal",
                "Up": {"Normalization": 0.1},
                "Down": {"Normalization": -0.05},
            }
        ]
    }
    sample = {"Name": "Signal"}
    region = {}
    # needs to be expanded to include histogram loading
    modifiers = workspace.get_sys_modifiers(config_example, region, sample, None)
    expected_modifiers = [
        {"name": "sys", "type": "normsys", "data": {"hi": 1.1, "lo": 0.95}}
    ]
    assert modifiers == expected_modifiers

    # unsupported systematics type
    config_example_unsupported = {
        "Systematics": [
            {"Name": "Normalization", "Type": "unknown", "Samples": "Signal"}
        ]
    }
    with pytest.raises(
        NotImplementedError, match="not supporting other systematic types yet"
    ):
        workspace.get_sys_modifiers(config_example_unsupported, region, sample, None)


@mock.patch("cabinetry.workspace.get_unc_for_sample", return_value=[0.1, 0.1])
@mock.patch(
    "cabinetry.workspace.get_yield_for_sample", return_value=[1.0, 2.0],
)
def test_get_channels(mock_get_yield, mock_get_unc):
    example_config = {
        "Regions": [{"Name": "region_1"}],
        "Samples": [{"Name": "signal"}, {"Data": True}],
        "NormFactors": [],
    }

    channels = workspace.get_channels(example_config, "path")
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
        ((example_config["Samples"][0], example_config["Regions"][0], "path"),)
    ]
    assert mock_get_unc.call_args_list == [
        ((example_config["Samples"][0], example_config["Regions"][0], "path"),)
    ]


def test_get_measurement():
    example_config = {
        "General": {"Measurement": "fit", "POI": "mu"},
        "NormFactors": [
            {"Name": "NF", "Nominal": 1.0, "Bounds": [0.0, 5.0], "Fixed": False}
        ],
    }
    expected_measurement = [
        {
            "name": "fit",
            "config": {
                "poi": "mu",
                "parameters": [
                    {
                        "name": "NF",
                        "inits": [1.0],
                        "bounds": [[0.0, 5.0]],
                        "fixed": False,
                    }
                ],
            },
        }
    ]
    assert workspace.get_measurements(example_config) == expected_measurement

    # no norm factor settings
    example_config_no_NF_settings = {
        "General": {"Measurement": "fit", "POI": "mu"},
        "NormFactors": [{"Name": "NF"}],
    }
    expected_measurement_no_NF_settings = [
        {"name": "fit", "config": {"poi": "mu", "parameters": [{"name": "NF"}]}}
    ]
    assert (
        workspace.get_measurements(example_config_no_NF_settings)
        == expected_measurement_no_NF_settings
    )


@mock.patch(
    "cabinetry.workspace.get_yield_for_sample", return_value=[1.0, 2.0],
)
def test_get_observations(mock_get_yield):
    # create observations list from config
    config = {
        "Samples": [{"Name": "data", "Data": True}],
        "Regions": [{"Name": "test_region"}],
    }
    obs = workspace.get_observations(config, "path")
    expected_obs = [{"name": "test_region", "data": [1.0, 2.0]}]
    assert obs == expected_obs
    assert mock_get_yield.call_args_list == [
        ((config["Samples"][0], config["Regions"][0], "path"),)
    ]


def test_build(tmp_path):
    minimal_config = {
        "General": {"Measurement": "test", "POI": "test_POI"},
        "Regions": [],
        "Samples": [{"Name": "data", "Data": True}],
    }
    ws = workspace.build(minimal_config, tmp_path, with_validation=False)
    ws_expected = {
        "channels": [],
        "measurements": [
            {"config": {"parameters": [], "poi": "test_POI"}, "name": "test"}
        ],
        "observations": [],
        "version": "1.0.0",
    }
    assert ws == ws_expected

    with mock.patch("cabinetry.workspace.validate") as mock_validate:
        ws = workspace.build(minimal_config, tmp_path, with_validation=True)
        assert ws == ws_expected
        assert mock_validate.call_args_list == [((ws,),)]


@mock.patch("cabinetry.workspace.pyhf.Workspace")
def test_validate(mock_validate):
    test_ws = {"workspace": "test"}
    workspace.validate(test_ws)
    assert mock_validate.call_args_list == [((test_ws,),)]


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
