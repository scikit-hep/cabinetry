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
            "HistogramFolder": "",
            "InputPath": "",
        },
        "Regions": [{"Name": "", "Filter": "", "Variable": "", "Binning": [0, 1]}],
        "Samples": [{"Name": "", "Tree": "", "Data": True}],
        "NormFactors": [{"Name": ""}],
    }
    assert configuration.validate(config_valid)

    # not exactly one data sample
    config_multiple_data_samples = {
        "General": {
            "Measurement": "",
            "HistogramFolder": "",
            "InputPath": "",
        },
        "Regions": [{"Name": "", "Filter": "", "Variable": "", "Binning": [0, 1]}],
        "Samples": [{"Name": "", "Tree": ""}],
        "NormFactors": [{"Name": ""}],
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

    # uniqueness of names
    with mock.patch("jsonschema.validate", return_value=None):
        config_region_names = {
            "Regions": [{"Name": "abc"}, {"Name": "abc"}],
            "Samples": [{"Name": "", "Tree": "", "Data": True}],
            "NormFactors": [],
        }
        with pytest.raises(ValueError, match="all region names must be unique"):
            configuration.validate(config_region_names)

        config_sample_names = {
            "Regions": [],
            "Samples": [{"Name": "abc", "Tree": "", "Data": True}, {"Name": "abc"}],
            "NormFactors": [],
        }
        with pytest.raises(ValueError, match="all sample names must be unique"):
            configuration.validate(config_sample_names)

        config_normfactor_names = {
            "Regions": [],
            "Samples": [{"Name": "", "Tree": "", "Data": True}],
            "NormFactors": [{"Name": "abc"}, {"Name": "abc"}],
        }
        with pytest.raises(ValueError, match="all normfactor names must be unique"):
            configuration.validate(config_normfactor_names)

        config_systematic_names = {
            "Regions": [],
            "Samples": [{"Name": "abc", "Tree": "", "Data": True}],
            "Systematics": [{"Name": "abc"}, {"Name": "abc"}],
            "NormFactors": [],
        }
        with pytest.raises(ValueError, match="all systematic names must be unique"):
            configuration.validate(config_systematic_names)


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
def test__setting_to_list(samples, converted):
    assert configuration._setting_to_list(samples) == converted


@pytest.mark.parametrize(
    "x_y_key, contained",
    [
        (({"Name": "abc"}, {}, "key"), True),
        (({"Name": "abc"}, {"key": ["abc", "def"]}, "key"), True),
        (({"Name": "abc"}, {"key": ["def"]}, "key"), False),
        (({"Name": "abc"}, {"key": "abc"}, "key"), True),
        (({"Name": "abc"}, {"key": "def"}, "key"), False),
    ],
)
def test__x_contains_y(x_y_key, contained):
    assert configuration._x_contains_y(*x_y_key) is contained


@pytest.mark.parametrize(
    "region_and_sample, contained",
    [
        (({"Name": "CR"}, {}), True),
        (({"Name": "SR"}, {"Regions": ["SR", "CR"]}), True),
        (({"Name": "CR"}, {"Regions": "SR"}), False),
    ],
)
def test_region_contains_sample(region_and_sample, contained):
    assert configuration.region_contains_sample(*region_and_sample) is contained


@pytest.mark.parametrize(
    "region_and_modifier, contained",
    [
        (({"Name": "CR"}, {}), True),
        (({"Name": "SR"}, {"Regions": ["SR", "CR"]}), True),
        (({"Name": "CR"}, {"Regions": "SR"}), False),
    ],
)
def test_region_contains_modifier(region_and_modifier, contained):
    assert configuration.region_contains_modifier(*region_and_modifier) is contained


@pytest.mark.parametrize(
    "sample_and_modifier, contained",
    [
        (({"Name": "Signal"}, {}), True),
        (({"Name": "Signal"}, {"Samples": ["Signal", "Background"]}), True),
        (({"Name": "Signal"}, {"Samples": "Background"}), False),
    ],
)
def test_sample_contains_modifier(sample_and_modifier, contained):
    assert configuration.sample_contains_modifier(*sample_and_modifier) is contained


@pytest.mark.parametrize(
    "reg_sam_sys_tem, is_needed",
    [
        # nominal
        (({}, {}, {"Name": "abc"}, None), True),
        # non-nominal data
        (({}, {"Data": True}, {"Name": "var"}, "Up"), False),
        # overall normalization variation
        (({}, {}, {"Type": "Normalization"}, "Up"), False),
        # normalization + shape variation
        (({}, {"Name": "Signal"}, {"Type": "NormPlusShape"}, "Up"), True),
        # normalization + shape variation on specified and affected sample
        (
            (
                {},
                {"Name": "Signal"},
                {"Type": "NormPlusShape", "Samples": ["Signal", "Background"]},
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
                "Up",
            ),
            False,
        ),
        # template not needed due to symmetrization
        (
            (
                {},
                {"Name": "Signal"},
                {"Type": "NormPlusShape", "Up": {"Symmetrize": True}},
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
        # region does not contain sample
        (
            (
                {"Name": "CR"},
                {"Name": "Signal", "Regions": "SR"},
                {"Type": "NormPlusShape"},
                None,
            ),
            False,
        ),
        # region does contain sample
        (
            (
                {"Name": "CR"},
                {"Name": "Signal", "Regions": "CR"},
                {"Type": "NormPlusShape"},
                None,
            ),
            True,
        ),
        # region does not contain modifier
        (
            (
                {"Name": "CR"},
                {"Name": "Signal"},
                {"Type": "NormPlusShape", "Regions": "SR"},
                "Up",
            ),
            False,
        ),
        # region does contain modifier
        (
            (
                {"Name": "CR"},
                {"Name": "Signal"},
                {"Type": "NormPlusShape", "Regions": "CR"},
                "Up",
            ),
            True,
        ),
    ],
)
def test_histogram_is_needed(reg_sam_sys_tem, is_needed):
    # could also mock region_contains_sample, region_contains_modifier, and
    # sample_contains_modifier
    reg, sam, sys, tem = reg_sam_sys_tem
    assert configuration.histogram_is_needed(*reg_sam_sys_tem) is is_needed


def test_histogram_is_needed_unknown():
    # non-supported systematic
    with pytest.raises(ValueError, match="unknown systematics type: unknown"):
        configuration.histogram_is_needed({}, {}, {"Type": "unknown"}, "Up")


def test_region_dict(caplog):
    caplog.set_level(logging.WARNING)

    config = {"Regions": [{"Name": "reg_a"}, {"Name": "reg_b"}]}
    assert configuration.region_dict(config, "reg_a") == {"Name": "reg_a"}

    config = {"Regions": [{"Name": "reg_a"}, {"Name": "reg_a"}]}
    assert configuration.region_dict(config, "reg_a") == {"Name": "reg_a"}
    assert "found more than one region with name reg_a" in [
        rec.message for rec in caplog.records
    ]

    with pytest.raises(ValueError, match="region abc not found in config"):
        configuration.region_dict(config, "abc")
