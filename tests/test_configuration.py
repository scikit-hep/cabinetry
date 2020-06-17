import pytest

from cabinetry import configuration


def test_read():
    configuration.read("config_example.yml")


def test_validate_missing_key():
    config_example = {"General": []}
    with pytest.raises(Exception) as e_info:
        configuration.validate(config_example)


def test_validate_unknown_key():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": "",
        "unknown": [],
    }
    with pytest.raises(Exception) as e_info:
        configuration.validate(config_example)


def test_validate_multiple_data_samples():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}, {"Data": True}],
    }
    with pytest.raises(Exception) as e_info:
        configuration.validate(config_example)


def test_validate_valid():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
    }
    configuration.validate(config_example)


def test_print_overview():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
        "Systematics": "",
    }
    configuration.print_overview(config_example)


def test_print_overview_no_sys():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
    }
    configuration.print_overview(config_example)


def test__convert_samples_to_list():
    assert configuration._convert_samples_to_list("sample") == ["sample"]
    assert configuration._convert_samples_to_list(["sample"]) == ["sample"]


def test_sample_affected_by_modifier():
    assert (
        configuration.sample_affected_by_modifier(
            {"Name": "Signal"}, {"Samples": ["Signal", "Background"]}
        )
        == True
    )
    assert (
        configuration.sample_affected_by_modifier(
            {"Name": "Signal"}, {"Samples": {"Background"}}
        )
        == False
    )


def test_histogram_is_needed():
    # nominal
    assert configuration.histogram_is_needed({}, {}, {"Name": "nominal"}) == True

    # non-nominal data
    assert (
        configuration.histogram_is_needed({"Data": True}, {}, {"Name": "var"}) == False
    )

    # overall normalization variation
    assert configuration.histogram_is_needed({}, {}, {"Type": "Overall"}) == False

    # normalization + shape variation on affected sample
    assert (
        configuration.histogram_is_needed(
            {"Name": "Signal"}, {}, {"Type": "NormPlusShape", "Samples": "Signal"}
        )
        == True
    )

    # normalization + shape variation on non-affected sample
    assert (
        configuration.histogram_is_needed(
            {"Name": "Background"}, {}, {"Type": "NormPlusShape", "Samples": "Signal"}
        )
        == False
    )

    # non-supported systematic
    with pytest.raises(Exception) as e_info:
        configuration.histogram_is_needed({}, {}, {"Type": "unknown"})
