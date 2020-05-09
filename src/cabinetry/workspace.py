import json
import logging
import os

from . import histo


log = logging.getLogger(__name__)


def _get_data_sample(config):
    """
    get the sample name of the data sample
    """
    data_samples = [sample for sample in config["Samples"] if sample.get("Data", False)]
    assert len(data_samples) == 1
    return data_samples[0]


def get_channels(config):
    """
    """
    return


def get_measurements(config):
    """
    """
    return


def get_observations(config, histogram_folder):
    """
    build the observations
    """
    data_sample = _get_data_sample(config)
    observations = []
    observation = {}
    for region in config["Regions"]:
        histogram_name = histo.build_name(data_sample, region, {"Name": "nominal"})
        histogram = histo.load(histogram_folder, histogram_name, modified=True)
        histo_yield = histogram["yields"].tolist()
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
    channels = get_channels(config)
    ws.update({"channels": channels})

    # measurements
    measurements = get_measurements(config)
    ws.update({"measurements": measurements})

    # build observations
    observations = get_observations(config, histogram_folder)
    ws.update({"observations": observations})

    # workspace version
    ws.update({"version": "1.0.0"})
    return ws


def save(ws, path, name):
    """
    save the workspace to a file
    """
    log.info("saving workspace %s to %s", name, path + name + ".json")

    # create output directory if it does not exist yet
    if not os.path.exists(path):
        os.mkdir(path)
    # save workspace to file
    with open(path + name + ".json", "w") as f:
        json.dump(ws, f, sort_keys=True, indent=4)
