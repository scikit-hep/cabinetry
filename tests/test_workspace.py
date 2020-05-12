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


def test_get_measurement():
    example_config = {"General": {"Measurement": "fit", "POI": "mu"}}
    expected_measurement = [{"name": "fit", "config": {"poi": "mu", "parameters": []}}]
    assert workspace.get_measurements(example_config) == expected_measurement
