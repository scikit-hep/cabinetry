"""Provides utilities to handle the cabinetry configuration."""

import json
import logging
import pathlib
import pkgutil
from typing import Any, Dict, List, Optional, Union

import jsonschema
import yaml

from cabinetry._typing import Literal


log = logging.getLogger(__name__)


def load(file_path_string: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads, validates, and returns a config file from the provided path.

    Args:
        file_path_string (Union[str, pathlib.Path]): path to config file

    Returns:
        Dict[str, Any]: cabinetry configuration
    """
    file_path = pathlib.Path(file_path_string)
    log.info(f"opening config file {file_path}")
    config = yaml.safe_load(file_path.read_text())
    validate(config)
    return config


def validate(config: Dict[str, Any]) -> bool:
    """Returns True if the config file is validated, otherwise raises exceptions.

    Checks that the config satisfies the json schema, and performs additional checks to
    validate the config further.

    Args:
        config (Dict[str, Any]): cabinetry configuration

    Raises:
        NotImplementedError: when more than one data sample is found
        ValueError: when region / sample / normfactor / systematic names are not unique

    Returns:
        bool: whether the validation was successful
    """
    # load json schema for config and validate against it
    schema_text = pkgutil.get_data(__name__, "schemas/config.json")
    if schema_text is None:
        raise FileNotFoundError("could not load config schema")
    config_schema = json.loads(schema_text)
    jsonschema.validate(instance=config, schema=config_schema)

    # check that there is exactly one data sample
    if sum(sample.get("Data", False) for sample in config["Samples"]) != 1:
        raise NotImplementedError("can only handle cases with exactly one data sample")

    # check that region names are unique
    region_names = [region["Name"] for region in config["Regions"]]
    if len(set(region_names)) != len(region_names):
        raise ValueError(f"all region names must be unique: {region_names}")

    # check that sample names are unique
    sample_names = [sample["Name"] for sample in config["Samples"]]
    if len(set(sample_names)) != len(sample_names):
        raise ValueError(f"all sample names must be unique: {sample_names}")

    # check that normfactor names are unique
    # may technically not be required, but non-unique names do not seem to offer any
    # obvious advantages, so require uniqueness here for consistency
    normfactor_names = [normfactor["Name"] for normfactor in config["NormFactors"]]
    if len(set(normfactor_names)) != len(normfactor_names):
        raise ValueError(f"all normfactor names must be unique: {normfactor_names}")

    # check that systematic names are unique
    # systematics are optional, may be empty
    systematic_names = [sys["Name"] for sys in config.get("Systematics", [])]
    if len(set(systematic_names)) != len(systematic_names):
        raise ValueError(f"all systematic names must be unique: {systematic_names}")

    # if no issues are found
    return True


def print_overview(config: Dict[str, Any]) -> None:
    """Prints a compact summary of a config file.

    Args:
        config (Dict[str, Any]): cabinetry configuration
    """
    log.info("the config contains:")
    log.info(f"  {len(config['Samples'])} Sample(s)")
    log.info(f"  {len(config['Regions'])} Regions(s)")
    log.info(f"  {len(config['NormFactors'])} NormFactor(s)")
    if "Systematics" in config.keys():
        log.info(f"  {len(config['Systematics'])} Systematic(s)")


def _setting_to_list(setting: Union[str, List[str]]) -> List[str]:
    """Converts a configuration setting to a list.

    The config allows for two ways of specifying some settings, for example samples. A
    single sample is specified as ``"Samples": "ABC"``, a list of samples as
    ``"Samples": ["ABC", "DEF"]``. For consistent treatment, the single sample is
    converted to a list.

    Args:
        setting (Union[str, List[str]]): name of single setting value or list of values

    Returns:
        list: name(s) of sample(s)
    """
    if not isinstance(setting, list):
        setting = [setting]
    return setting


def _x_contains_y(x: Dict[str, Any], y: Dict[str, Any], y_key: str) -> bool:
    """Checks if object ``x`` contains ``y`` using property ``y_key`` of ``y``.

    If ``y_key`` is not specified, ``x`` is assumed to contain ``y`` by default. Used
    to check if regions contain samples/modifiers and if samples contain modifiers.
    ``x`` is identified by its "Name" property, which must exist.

    Args:
        x (Dict[str, Any]): containing all relevant information: region or sample, must
            have a "Name" property
        y (Dict[str, Any]): containing all relevant information: sample or modifier
        y_key (str): property of ``y`` to check

    Returns:
        bool: True if ``x`` contains ``y``, False otherwise
    """
    # y_key setting of y is optional, default to empty list
    matched_x_list = _setting_to_list(y.get(y_key, []))
    if matched_x_list and x["Name"] not in matched_x_list:
        # only some x are allowed as specified in list, and current x does not match
        return False
    return True


def region_contains_sample(region: Dict[str, Any], sample: Dict[str, Any]) -> bool:
    """Checks if a region contains a given sample.

    A sample enters all regions by default, and its "Regions" property can be used to
    specify a single region or list of regions that contain the sample.

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information

    Returns:
        bool: True if region contains sample, False otherwise
    """
    return _x_contains_y(region, sample, "Regions")


def region_contains_modifier(region: Dict[str, Any], modifier: Dict[str, Any]) -> bool:
    """Checks if a region contains a given modifier (Systematic, NormFactor).

    A modifier affects all regions by default, and its "Regions" property can be used to
    specify a single region or list of regions that contain the modifier. This does not
    check whether the modifier only acts on samples which the region does not contain.

    Args:
        region (Dict[str, Any]): containing all region information
        modifier (Dict[str, Any]): containing all modifier information (a Systematic or
            a NormFactor)

    Returns:
        bool: True if region contains modifier, False otherwise
    """
    return _x_contains_y(region, modifier, "Regions")


def sample_contains_modifier(sample: Dict[str, Any], modifier: Dict[str, Any]) -> bool:
    """Checks if a sample is affected by a given modifier (Systematic, NormFactor).

    A modifier affects all samples by default, and its "Samples" property can be used to
    specify a single sample or list of samples on which the modifier acts.

    Args:
        sample (Dict[str, Any]): containing all sample information
        modifier (Dict[str, Any]): containing all modifier information (a Systematic or
            a NormFactor)

    Returns:
        bool: True if sample is affected, False otherwise
    """
    return _x_contains_y(sample, modifier, "Samples")


def histogram_is_needed(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> bool:
    """Determines whether a histogram is needed for a specific configuration.

    The configuration is defined by the region, sample, systematic and template ("Up" or
    "Down", None for nominal).

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): which template to consider: "Up",
            "Down", None for the nominal case

    Raises:
        NotImplementedError: non-supported systematic variations based on histograms are
            requested

    Returns:
        bool: whether a histogram is needed
    """
    if not region_contains_sample(region, sample):
        # region does not contain sample
        return False

    if template is None:
        # for nominal, histograms are generally needed
        histo_needed = True
    else:
        # non-nominal case
        if sample.get("Data", False):
            # for data, only nominal histograms are needed
            histo_needed = False
        else:
            # handle non-nominal, non-data histograms
            # this assumes that the systematic dict satisfies config schema requirements
            if systematic["Type"] == "Normalization":
                # no histogram needed for normalization variation
                histo_needed = False
            elif systematic["Type"] == "NormPlusShape":
                # for a variation defined via a template, a histogram is needed (if
                # sample is affected in region)
                histo_needed = region_contains_modifier(region, systematic)
                histo_needed &= sample_contains_modifier(sample, systematic)
                # if symmetrization is specified for the template under consideration,
                # a histogram is not needed (since it will later on be obtained via
                # symmetrization)
                if systematic.get(template, {}).get("Symmetrize", False):
                    histo_needed = False
            else:
                raise ValueError(f"unknown systematics type: {systematic['Type']}")

    return histo_needed


def region_dict(config: Dict[str, Any], region_name: str) -> Dict[str, Any]:
    """Returns the dictionary for a region with the given name.

    Args:
        config (Dict[str, Any]): cabinetry configuration file
        region_name (str): name of region

    Raises:
        ValueError: when region is not found in config

    Returns:
        Dict[str, Any]: dictionary describing region
    """
    regions = [reg for reg in config["Regions"] if reg["Name"] == region_name]
    if len(regions) == 0:
        raise ValueError(f"region {region_name} not found in config")
    if len(regions) > 1:
        log.error(f"found more than one region with name {region_name}")
    return regions[0]
