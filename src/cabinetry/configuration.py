import logging
from pathlib import Path
import yaml


REQUIRED_CONFIG_KEYS = ["General", "Samples", "Regions", "NormFactors"]

OPTIONAL_CONFIG_KEYS = ["Systematics"]

log = logging.getLogger(__name__)


def read(file_path_string):
    """read a config file from a provided path and return it

    Args:
        file_path_string (str): path to config file

    Returns:
        dict: cabinetry configuration
    """
    file_path = Path(file_path_string)
    log.info("opening config file %s", file_path)
    config = yaml.safe_load(file_path.read_text())
    validate(config)
    return config


def validate(config):
    """test whether the config is valid

    Args:
        config (dict): cabinetry configuration

    Raises:
        ValueError: when missing required keys
        ValueError: when unknown keys are found
        NotImplementedError: when more than one data sample is found
    """
    config_keys = config.keys()

    # check whether all required keys exist
    for required_key in REQUIRED_CONFIG_KEYS:
        if required_key not in config_keys:
            raise ValueError("missing required key in config:", required_key)

    # check whether all keys are known
    for key in config_keys:
        if key not in (REQUIRED_CONFIG_KEYS + OPTIONAL_CONFIG_KEYS):
            raise ValueError("unknown key found:", key)

    # check that there is exactly one data sample
    if sum([sample.get("Data", False) for sample in config["Samples"]]) != 1:
        raise NotImplementedError("can only handle cases with exactly one data sample")

    # should also check here for conflicting settings

    # if no issues are found
    return True


def print_overview(config):
    """output a compact summary of a config file

    Args:
        config (dict): cabinetry configuration
    """
    log.info("the config contains:")
    log.info("  %i Sample(s)", len(config["Samples"]))
    log.info("  %i Regions(s)", len(config["Regions"]))
    log.info("  %i NormFactor(s)", len(config["NormFactors"]))
    if "Systematics" in config.keys():
        log.info("  %i Systematic(s)", len(config["Systematics"]))


def _convert_samples_to_list(samples):
    """the config can allow for two ways of specifying samples, a single sample:
    "Samples": "ABC"
    or a list of samples:
    "Samples": ["ABC", "DEF"]
    for consistent treatment, convert the single sample into a single-element list

    Args:
        samples (string/list): name of single sample or list of sample names

    Returns:
        list: name(s) of sample(s)
    """
    if not isinstance(samples, list):
        samples = [samples]
    return samples


def sample_affected_by_modifier(sample, modifier):
    """check if a sample is affected by a given modifier (Systematic, NormFactor)

    Args:
        sample (dict): containing all sample information
        modifier (dict): containing all modifier information (a Systematic of a NormFactor)

    Returns:
        bool: True if sample is affected, False otherwise
    """
    affected_samples = _convert_samples_to_list(modifier["Samples"])
    affected = sample["Name"] in affected_samples
    return affected


def histogram_is_needed(sample, region, systematic):
    """determine whether for a given sample-region-systematic pairing, there is
    an associated histogram

    Args:
        sample (dict): containing all sample information
        region (dict): containing all region information
        systematic (dict): containing all systematic information

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
            if systematic["Type"] == "Overall":
                # no histogram needed for normalization variation
                histo_needed = False
            elif systematic["Type"] == "NormPlusShape":
                # for a variation defined via a template, a histogram is needed (if sample is affected)
                histo_needed = sample_affected_by_modifier(sample, systematic)
            else:
                raise NotImplementedError("other systematics not yet implemented")

    return histo_needed
