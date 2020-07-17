import numpy as np
import pytest

from cabinetry import workspace
from cabinetry import histo


def test__get_data_sample():
    mc_sample = {"Name": "MC"}
    data_sample = {"Name": "Data", "Data": True}
    config_example = {"Samples": [mc_sample, data_sample]}
    assert workspace._get_data_sample(config_example) == data_sample


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


def test_get_OverallSys_modifier():
    systematic = {"Name": "sys", "OverallUp": 0.1, "OverallDown": -0.05}
    expected_modifier = {
        "name": "sys",
        "type": "normsys",
        "data": {"hi": 1.1, "lo": 0.95},
    }
    assert workspace.get_OverallSys_modifier(systematic) == expected_modifier


def test_get_sys_modifiers():
    config_example = {
        "Systematics": [
            {
                "Name": "sys",
                "Type": "Overall",
                "Samples": "Signal",
                "OverallUp": 0.1,
                "OverallDown": -0.05,
            }
        ]
    }
    sample = {"Name": "Signal"}
    region = {}
    # needs to be expanded to include histogram loading
    modifiers = workspace.get_sys_modifiers(config_example, sample, region, None)
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
    ) as e_info:
        workspace.get_sys_modifiers(config_example_unsupported, sample, region, None)


def test_get_channels(tmp_path):
    example_config = {
        "Regions": [{"Name": "region_1"}],
        "Samples": [{"Name": "signal"}],
        "NormFactors": [],
    }

    # create a histogram for testing
    histo_path = tmp_path / "region_1_signal_nominal.npz"
    histogram = histo.Histogram.from_arrays([0.0, 1.0, 2.0], [1.0, 2.0], [1.0, 1.0])
    histogram.save(histo_path)

    channels = workspace.get_channels(example_config, tmp_path)
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
                            "data": [1.0, 1.0],
                        }
                    ],
                }
            ],
        }
    ]
    assert channels == expected_channels


def test_get_measurement():
    example_config = {"General": {"Measurement": "fit", "POI": "mu"}}
    expected_measurement = [{"name": "fit", "config": {"poi": "mu", "parameters": []}}]
    assert workspace.get_measurements(example_config) == expected_measurement


def test_get_observations(tmp_path):
    histo_path = tmp_path / "test_region_Data_nominal.npz"

    # build a test histogram and save it
    histogram = histo.Histogram.from_arrays([0.0, 1.0, 2.0], [1.0, 2.0], [1.0, 1.0])
    histogram.save(histo_path)

    # create observations list from config
    config = {
        "Samples": [{"Name": "Data", "Tree": "tree", "Path": tmp_path, "Data": True}],
        "Regions": [{"Name": "test_region"}],
    }
    obs = workspace.get_observations(config, tmp_path)
    expected_obs = [{"name": "test_region", "data": [1.0, 2.0]}]
    assert obs == expected_obs


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
