import logging
from unittest import mock

import jsonschema
import pytest

from cabinetry import configuration


@mock.patch("cabinetry.configuration.validate")
def test_load(mock_validation):
    conf = configuration.load("config_example.yml")
    assert isinstance(conf, dict)
    mock_validation.assert_called_once()


def test_validate():
    config_valid = {
        "General": {
            "Measurement": "",
            "POI": "",
            "HistogramFolder": "",
            "InputPath": "",
        },
        "Regions": [{"Name": "", "Filter": "", "Variable": "", "Binning": [0, 1]}],
        "Samples": [{"Name": "", "Tree": "", "Data": True}],
        "NormFactors": [{"Name": "", "Samples": ""}],
    }
    assert configuration.validate(config_valid)

    # not exactly one data sample
    config_multiple_data_samples = {
        "General": {
            "Measurement": "",
            "POI": "",
            "HistogramFolder": "",
            "InputPath": "",
        },
        "Regions": [{"Name": "", "Filter": "", "Variable": "", "Binning": [0, 1]}],
        "Samples": [{"Name": "", "Tree": ""}],
        "NormFactors": [{"Name": "", "Samples": ""}],
    }
    with pytest.raises(
        NotImplementedError, match="can only handle cases with exactly one data sample"
    ):
        configuration.validate(config_multiple_data_samples)

    # config doesn't adhere to schema
    config_schema_mismatch = {}
    with pytest.raises(
        jsonschema.exceptions.ValidationError, match="'General' is a required property"
    ):
        configuration.validate(config_schema_mismatch)

    # schema cannot be loaded
    with mock.patch(
        "cabinetry.configuration.pkgutil.get_data", return_value=None
    ) as mock_get:
        with pytest.raises(FileNotFoundError, match="could not load config schema"):
            configuration.validate({})
        mock_get.assert_called_once()


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
def test__convert_setting_to_list(samples, converted):
    assert configuration._convert_setting_to_list(samples) == converted


@pytest.mark.parametrize(
    "sample_and_modifier, affected",
    [
        (({"Name": "Signal"}, {"Samples": ["Signal", "Background"]}), True),
        (({"Name": "Signal"}, {"Samples": "Background"}), False),
    ],
)
def test_sample_affected_by_modifier(sample_and_modifier, affected):
    assert configuration.sample_affected_by_modifier(*sample_and_modifier) is affected


@pytest.mark.parametrize(
    "reg_sam_sys_tem, is_needed",
    [
        # nominal
        (({}, {}, {"Name": "Nominal"}, "Nominal"), True),
        # non-nominal data
        (({}, {"Data": True}, {"Name": "var"}, ""), False),
        # overall normalization variation
        (({}, {}, {"Type": "Normalization"}, ""), False),
        # normalization + shape variation on affected sample
        (
            (
                {},
                {"Name": "Signal"},
                {"Type": "NormPlusShape", "Samples": "Signal"},
                "Up",
            ),
            True,
        ),
        # normalization + shape variation on non-affected sample
        (
            (
                {},
                {"Name": "Background"},
                {"Type": "NormPlusShape", "Samples": "Signal"},
                "",
            ),
            False,
        ),
        # non-needed template of systematic due to symmetrization
        (
            (
                {},
                {"Name": "Signal"},
                {
                    "Type": "NormPlusShape",
                    "Samples": "Signal",
                    "Up": {"Symmetrize": True},
                },
                "Up",
            ),
            False,
        ),
        # template needed since symmetrization is only for other direction
        (
            (
                {},
                {"Name": "Signal"},
                {
                    "Type": "NormPlusShape",
                    "Samples": "Signal",
                    "Down": {"Symmetrize": True},
                },
                "Up",
            ),
            True,
        ),
    ],
)
def test_histogram_is_needed(reg_sam_sys_tem, is_needed):
    reg, sam, sys, tem = reg_sam_sys_tem
    assert configuration.histogram_is_needed(*reg_sam_sys_tem) is is_needed


def test_histogram_is_needed_unknown():
    # non-supported systematic
    with pytest.raises(
        NotImplementedError, match="other systematics not yet implemented"
    ):
        configuration.histogram_is_needed({}, {}, {"Type": "unknown"}, "")


def test_get_region_dict(caplog):
    config = {"Regions": [{"Name": "reg_a"}, {"Name": "reg_b"}]}
    assert configuration.get_region_dict(config, "reg_a") == {"Name": "reg_a"}

    config = {"Regions": [{"Name": "reg_a"}, {"Name": "reg_a"}]}
    assert configuration.get_region_dict(config, "reg_a") == {"Name": "reg_a"}
    assert "found more than one region with name reg_a" in [
        rec.message for rec in caplog.records
    ]

    with pytest.raises(ValueError, match="region abc not found in config"):
        configuration.get_region_dict(config, "abc")
