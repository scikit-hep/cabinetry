import logging
from unittest import mock

import pytest

from cabinetry import configuration


@mock.patch("cabinetry.configuration.validate")
def test_read(mock_validation):
    conf = configuration.read("config_example.yml")
    assert isinstance(conf, dict)
    mock_validation.assert_called_once()


def test_validate():
    config_valid = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
    }
    assert configuration.validate(config_valid)

    config_missing_key = {"General": []}
    with pytest.raises(ValueError, match="missing required key in config") as e_info:
        configuration.validate(config_missing_key)

    config_unknown_key = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": "",
        "unknown": [],
    }
    with pytest.raises(ValueError, match="unknown key found") as e_info:
        configuration.validate(config_unknown_key)

    config_multiple_data_samples = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}, {"Data": True}],
    }
    with pytest.raises(
        NotImplementedError, match="can only handle cases with exactly one data sample"
    ) as e_info:
        configuration.validate(config_multiple_data_samples)


def test_print_overview(caplog):
    caplog.set_level(logging.DEBUG)
    config_example = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
        "Systematics": "",
    }
    configuration.print_overview(config_example)
    assert "  1 Sample(s)" in [rec.message for rec in caplog.records]
    assert "  0 Regions(s)" in [rec.message for rec in caplog.records]
    assert "  0 NormFactor(s)" in [rec.message for rec in caplog.records]
    assert "  0 Systematic(s)" in [rec.message for rec in caplog.records]
    caplog.clear()

    config_example_no_sys = {
        "General": "",
        "Regions": "",
        "NormFactors": "",
        "Samples": [{"Data": True}],
    }
    configuration.print_overview(config_example_no_sys)
    assert "Systematic(s)" not in [rec.message for rec in caplog.records]
    caplog.clear()


@pytest.mark.parametrize(
    "samples, converted", [("sample", ["sample"]), (["sample"], ["sample"])]
)
def test__convert_samples_to_list(samples, converted):
    assert configuration._convert_samples_to_list(samples) == converted


@pytest.mark.parametrize(
    "sample_and_modifier, affected",
    [
        (({"Name": "Signal"}, {"Samples": ["Signal", "Background"]}), True),
        (({"Name": "Signal"}, {"Samples": {"Background"}}), False),
    ],
)
def test_sample_affected_by_modifier(sample_and_modifier, affected):
    assert configuration.sample_affected_by_modifier(*sample_and_modifier) is affected


@pytest.mark.parametrize(
    "reg_sam_sys, needed",
    [
        # nominal
        (({}, {}, {"Name": "nominal"}), True),
        # non-nominal data
        (({}, {"Data": True}, {"Name": "var"}), False),
        # overall normalization variation
        (({}, {}, {"Type": "Overall"}), False),
        # normalization + shape variation on affected sample
        (
            ({}, {"Name": "Signal"}, {"Type": "NormPlusShape", "Samples": "Signal"}),
            True,
        ),
        # normalization + shape variation on non-affected sample
        (
            (
                {},
                {"Name": "Background"},
                {"Type": "NormPlusShape", "Samples": "Signal"},
            ),
            False,
        ),
    ],
)
def test_histogram_is_needed(reg_sam_sys, needed):
    assert configuration.histogram_is_needed(*reg_sam_sys) is needed

    # non-supported systematic
    with pytest.raises(
        NotImplementedError, match="other systematics not yet implemented"
    ) as e_info:
        configuration.histogram_is_needed({}, {}, {"Type": "unknown"})
