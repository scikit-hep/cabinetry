import json
import logging
import pathlib
import pkgutil
from typing import Any, Dict, List, Union

import jsonschema
import yaml

log = logging.getLogger(__name__)


def read(file_path_string: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """read a config file from a provided path and return it

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
    """check whether the config satisfies the json schema, and perform additional
    checks to validate it

    Args:
        config (Dict[str, Any]): cabinetry configuration

    Raises:
        ValueError: when missing required keys
        ValueError: when unknown keys are found
        NotImplementedError: when more than one data sample is found
        ValueError: when missing a name for a NormFactor
        ValueError: when missing a samples for a NormFactor

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
    if sum([sample.get("Data", False) for sample in config["Samples"]]) != 1:
        raise NotImplementedError("can only handle cases with exactly one data sample")

    # should also check here for conflicting settings
    ...

    # if no issues are found
    return True


def print_overview(config: Dict[str, Any]) -> None:
    """output a compact summary of a config file

    Args:
        config (Dict[str, Any]): cabinetry configuration
    """
    log.info("the config contains:")
    log.info(f"  {len(config['Samples'])} Sample(s)")
    log.info(f"  {len(config['Regions'])} Regions(s)")
    log.info(f"  {len(config['NormFactors'])} NormFactor(s)")
    if "Systematics" in config.keys():
        log.info(f"  {len(config['Systematics'])} Systematic(s)")


def _convert_samples_to_list(samples: Union[str, List[str]]) -> List[str]:
    """the config can allow for two ways of specifying samples, a single sample:
    "Samples": "ABC"
    or a list of samples:
    "Samples": ["ABC", "DEF"]
    for consistent treatment, convert the single sample into a single-element list

    Args:
        samples (Union[str, List[str]]): name of single sample or list of sample names

    Returns:
        list: name(s) of sample(s)
    """
    if not isinstance(samples, list):
        samples = [samples]
    return samples


def sample_affected_by_modifier(
    sample: Dict[str, Any], modifier: Dict[str, Any]
) -> bool:
    """check if a sample is affected by a given modifier (Systematic, NormFactor)

    Args:
        sample (Dict[str, Any]): containing all sample information
        modifier (Dict[str, Any]): containing all modifier information (a Systematic of a NormFactor)

    Returns:
        bool: True if sample is affected, False otherwise
    """
    affected_samples = _convert_samples_to_list(modifier["Samples"])
    affected = sample["Name"] in affected_samples
    return affected


def histogram_is_needed(
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: str,
) -> bool:
    """determine whether for a given sample-region-systematic pairing, there is
    an associated histogram

    Args:
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (str): which template is considered: "Nominal", "Up", "Down"

    Raises:
        NotImplementedError: non-supported systematic variations based on histograms are requested

    Returns:
        bool: whether a histogram is needed
    """
    if systematic.get("Name", None) == "nominal":
        # for nominal, histograms are generally needed
        histo_needed = True
    else:
        # non-nominal case
        if sample.get("Data", False):
            # for data, only nominal histograms are needed
            histo_needed = False
        else:
            # handle non-nominal, non-data histograms
            if systematic["Type"] == "Normalization":
                # no histogram needed for normalization variation
                histo_needed = False
            elif systematic["Type"] == "NormPlusShape":
                # for a variation defined via a template, a histogram is needed (if sample is affected)
                histo_needed = sample_affected_by_modifier(sample, systematic)
                # if symmetrization is specified for the template under consideration, a histogram
                # is not needed (since it will later on be obtained via symmetrization)
                if systematic.get(template, {}).get("Symmetrize", False):
                    histo_needed = False
            else:
                raise NotImplementedError("other systematics not yet implemented")

    return histo_needed
