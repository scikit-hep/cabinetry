import logging
import yaml


REQUIRED_CONFIG_KEYS = ["General", "Samples", "Regions", "NormFactors"]

OPTIONAL_CONFIG_KEYS = ["Systematics"]

log = logging.getLogger(__name__)


def read(file_path):
    """
    read a config file from a provided path and return it
    """
    log.info("opening config file %s", file_path)
    with open(file_path) as f:
        config = yaml.safe_load(f)
    validate(config)
    return config


def validate(config):
    """
    test whether the config is valid
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
    pass


def print_overview(config):
    """
    output a compact summary of a config file
    """
    log.info("the config contains:")
    log.info("  %i Sample(s)", len(config["Samples"]))
    log.info("  %i Regions(s)", len(config["Regions"]))
    log.info("  %i NormFactor(s)", len(config["NormFactors"]))
    if "Systematics" in config.keys():
        log.info("  %i Systematic(s)", len(config["Systematics"]))
