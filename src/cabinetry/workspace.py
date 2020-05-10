import json
import logging
import os

import pyhf

from . import histo


log = logging.getLogger(__name__)


def _get_data_sample(config):
    """
    get the sample name of the data sample
    """
    data_samples = [sample for sample in config["Samples"] if sample.get("Data", False)]
    assert len(data_samples) == 1
    return data_samples[0]


def get_yield_for_sample(
    sample, region, histogram_folder, systematic={"Name": "nominal"}
):
    """
    get the yield for a specific sample, by figuring out its name and then
    obtaining the yield from the correct histogram
    """
    histogram_name = histo.build_name(sample, region, systematic)
    histogram = histo.load(histogram_folder, histogram_name, modified=True)
    histo_yield = histogram["yields"].tolist()
    return histo_yield


def get_unc_for_sample(
    sample, region, histogram_folder, systematic={"Name": "nominal"}
):
    """
    get the uncertainty of a specific sample, by figuring out its name and then
    obtaining the sumw2 from the correct histogram
    """
    histogram_name = histo.build_name(sample, region, systematic)
    histogram = histo.load(histogram_folder, histogram_name, modified=True)
    histo_yield = histogram["sumw2"].tolist()
    return histo_yield


def get_NF_modifiers(config, sample):
    """
    get the list of NormFactor modifiers acting on a sample
    """
    modifiers = []
    for NormFactor in config["NormFactors"]:
        # if a single sample is affected by the normfactor, put it in a list as a single entry
        # could force that behavior in config syntax instead by requiring a single sample to be
        # still listed as "Samples": ["sample_name"]
        affected_samples = (
            NormFactor["Samples"]
            if isinstance(NormFactor["Samples"], list)
            else [NormFactor["Samples"]]
        )
        if sample["Name"] in affected_samples:
            log.debug(
                "adding NormFactor %s to sample %s", NormFactor["Name"], sample["Name"]
            )
            modifiers.append(
                {"data": None, "name": NormFactor["Name"], "type": "normfactor"}
            )
    return modifiers


def get_channels(config, histogram_folder):
    """
    construct the channel information: yields per sample and modifiers
    """
    channels = []
    for region in config["Regions"]:
        channel = {}
        channel.update({"name": region["Name"]})
        samples = []
        for sample in config["Samples"]:
            if sample.get("Data", False):
                # skip the data sample, it goes into the observations instead
                continue

            # yield of the samples
            histo_yield = get_yield_for_sample(sample, region, histogram_folder)
            current_sample = {}
            current_sample.update({"name": sample["Name"]})
            current_sample.update({"data": histo_yield})

            # collect all modifiers for the sample
            modifiers = []

            # gammas
            stat_unc = get_unc_for_sample(sample, region, histogram_folder)
            gammas = {}
            gammas.update({"name": "staterror_" + region["Name"].replace(" ", "-")})
            gammas.update({"type": "staterror"})
            gammas.update({"data": stat_unc})
            modifiers.append(gammas)

            # check if Normfactor affect the sample and add modifiers as needed
            NF_modifier_list = get_NF_modifiers(config, sample)
            modifiers += NF_modifier_list

            current_sample.update({"modifiers": modifiers})
            samples.append(current_sample)
        channel.update({"samples": samples})
        channels.append(channel)
    return channels


def get_measurements(config):
    """
    construct the measurements, including POI setting and lumi
    only supporting a single measurement so far
    """
    measurements = []
    measurement = {}
    measurement.update({"name": config["General"]["Measurement"]})
    config_dict = {}
    parameters = {"parameters": []}
    config_dict.update(parameters)
    config_dict.update({"poi": config["General"]["POI"]})
    measurement.update({"config": config_dict})
    measurements.append(measurement)
    return measurements


def get_observations(config, histogram_folder):
    """
    build the observations
    """
    data_sample = _get_data_sample(config)
    observations = []
    observation = {}
    for region in config["Regions"]:
        histo_yield = get_yield_for_sample(data_sample, region, histogram_folder)
        observation.update({"name": region["Name"]})
        observation.update({"data": histo_yield})
        observations.append(observation)
    return observations


def build(config, histogram_folder):
    """
    build a HistFactory workspace, pyhf style
    """
    log.info("building workspace")

    ws = {}  # the workspace

    # channels
    channels = get_channels(config, histogram_folder)
    ws.update({"channels": channels})

    # measurements
    measurements = get_measurements(config)
    ws.update({"measurements": measurements})

    # build observations
    observations = get_observations(config, histogram_folder)
    ws.update({"observations": observations})

    # workspace version
    ws.update({"version": "1.0.0"})

    # validate the workspace
    validate(ws)
    return ws


def validate(ws):
    """
    validate a workspace
    """
    pyhf.Workspace(ws)


def save(ws, path, name):
    """
    save the workspace to a file
    """
    log.debug("saving workspace %s to %s", name, path + name + ".json")

    # create output directory if it does not exist yet
    if not os.path.exists(path):
        os.mkdir(path)
    # save workspace to file
    with open(path + name + ".json", "w") as f:
        json.dump(ws, f, sort_keys=True, indent=4)
