import json
import logging
import os
from pathlib import Path

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
    histogram, _ = histo.load_from_config(
        histogram_folder, sample, region, systematic, modified=True
    )
    histo_yield = histogram["yields"].tolist()
    return histo_yield


def get_unc_for_sample(
    sample, region, histogram_folder, systematic={"Name": "nominal"}
):
    """
    get the uncertainty of a specific sample, by figuring out its name and then
    obtaining the sumw2 from the correct histogram
    """
    histogram, _ = histo.load_from_config(
        histogram_folder, sample, region, systematic, modified=True
    )
    histo_yield = histogram["sumw2"].tolist()
    return histo_yield


def _convert_samples_to_list(samples):
    """
    the config can allow for two ways of specifying samples, a single sample:
    "Samples": "ABC"
    or a list of samples:
    "Samples": ["ABC", "DEF"]
    for consistent treatment, convert the single sample into a single-element list
    """
    if isinstance(samples, list):
        return samples
    else:
        return [samples]


def get_NF_modifiers(config, sample):
    """
    get the list of NormFactor modifiers acting on a sample
    """
    modifiers = []
    for NormFactor in config["NormFactors"]:
        affected_samples = _convert_samples_to_list(NormFactor["Samples"])

        if sample["Name"] in affected_samples:
            log.debug(
                "adding NormFactor %s to sample %s", NormFactor["Name"], sample["Name"]
            )
            modifiers.append(
                {"data": None, "name": NormFactor["Name"], "type": "normfactor"}
            )
    return modifiers


def get_OverallSys_modifier(systematic):
    """
    construct an OverallSys modifier
    while this can be built without any histogram reference, it might be useful
    to build a histogram for this anyway and possibly use it here
    """
    modifier = {}
    modifier.update({"name": systematic["Name"]})
    modifier.update({"type": "normsys"})
    modifier.update(
        {
            "data": {
                "hi": 1 + systematic["OverallUp"],
                "lo": 1 + systematic["OverallDown"],
            }
        }
    )
    return modifier


def get_sys_modifiers(config, sample):
    """
    get the list of all systematic modifiers acting on a sample
    """
    modifiers = []
    for systematic in config["Systematics"]:
        if systematic["Type"] == "OVERALL":
            # OverallSys (norm uncertainty with Gaussian constraint)
            log.debug(
                "adding OverallSys %s to sample %s", systematic["Name"], sample["Name"]
            )
            modifiers.append(get_OverallSys_modifier(systematic))
        else:
            raise NotImplementedError("not supporting other systematic types yet")
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

            # check if systematic uncertainties affect the samples, add modifiers as needed
            sys_modifier_list = get_sys_modifiers(config, sample)
            modifiers += sys_modifier_list

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


def save(ws, file_path_string):
    """
    save the workspace to a file
    """
    file_path = Path(file_path_string)
    log.debug("saving workspace to %s", file_path)
    # create output directory if it does not exist yet
    if not os.path.exists(file_path.parent):
        os.mkdir(file_path.parent)
    # save workspace to file
    file_path.write_text(json.dumps(ws, sort_keys=True, indent=4))


def load(file_path_string):
    """
    load a workspace from file
    """
    file_path = Path(file_path_string)
    ws = json.loads(file_path.read_text())
    return ws
