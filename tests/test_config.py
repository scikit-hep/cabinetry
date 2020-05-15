import pytest

from cabinetry import config


def test_read():
    config.read("config_example.yml")


def test_validate_missing_key():
    config_example = {"General": []}
    with pytest.raises(Exception) as e_info:
        config.validate(config_example)


def test_validate_unknown_key():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": "",
        "unknown": [],
    }
    with pytest.raises(Exception) as e_info:
        config.validate(config_example)


def test_validate_multiple_data_samples():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}, {"Data": True}],
    }
    with pytest.raises(Exception) as e_info:
        config.validate(config_example)


def test_validate_valid():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
    }
    config.validate(config_example)


def test_print_overview():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
        "Systematics": "",
    }
    config.print_overview(config_example)


def test_print_overview_no_sys():
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
    }
    config.print_overview(config_example)
