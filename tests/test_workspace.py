import pytest

from cabinetry import workspace


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
        "Systematics": [{"Name": "Normalization", "Type": "unknown"}]
    }
    with pytest.raises(Exception) as e_info:
        workspace.get_sys_modifiers(config_example_unsupported, sample, region, None)


def test_get_measurement():
    example_config = {"General": {"Measurement": "fit", "POI": "mu"}}
    expected_measurement = [{"name": "fit", "config": {"poi": "mu", "parameters": []}}]
    assert workspace.get_measurements(example_config) == expected_measurement
