import yaml

REQUIRED_CONFIG_KEYS = ["General", "Samples", "Regions", "NormFactors"]

OPTIONAL_CONFIG_KEYS = ["Systematics"]


def read(file_path):
    """
    read a config file from a provided path and return it
    """
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

    # should also check here for conflicting settings
    pass


def print_overview(config):
    """
    output a compact summary of a config file
    """
    print("# the config contains:")
    print("-", len(config["Samples"]), "Sample(s)")
    print("-", len(config["Regions"]), "Region(s)")
    print("-", len(config["NormFactors"]), "NormFactor(s)")
    if "Systematics" in config.keys():
        print("-", len(config["Systematics"]), "Systematic(s)")
