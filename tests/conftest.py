import logging.config

import boost_histogram as bh
import numpy as np
import pytest
import uproot


def pytest_sessionstart():
    # suppress verbose DEBUG level output from matplotlib when tests fail
    logging.config.dictConfig(
        {
            "loggers": {"matplotlib": {"level": "INFO"}},
            "disable_existing_loggers": False,
            "version": 1,
        }
    )


class Utils:
    @staticmethod
    def create_ntuple(fname, treename, varname, var_array, weightname, weight_array):
        with uproot.recreate(fname) as f:
            f[treename] = {varname: var_array, weightname: weight_array}

    @staticmethod
    def create_histogram(fname, histname, bins, yields, stdev):
        hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
        hist[...] = np.stack([yields, stdev**2], axis=-1)
        with uproot.recreate(fname) as f:
            f[histname] = hist


@pytest.fixture
def utils():
    return Utils


@pytest.fixture
def example_spec():
    spec = {
        "channels": [
            {
                "name": "Signal Region",
                "samples": [
                    {
                        "data": [51.8],
                        "modifiers": [
                            {
                                "data": [2.6],
                                "name": "staterror_Signal-Region",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {
                            "name": "staterror_Signal-Region",
                            "fixed": True,
                            "inits": [1.1],
                        }
                    ],
                    "poi": "Signal strength",
                },
                "name": "My fit",
            }
        ],
        "observations": [{"data": [475], "name": "Signal Region"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_multibin():
    spec = {
        "channels": [
            {
                "name": "region_1",
                "samples": [
                    {
                        "data": [25, 5],
                        "modifiers": [
                            {
                                "data": [5, 2],
                                "name": "staterror_region_1",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            },
            {
                "name": "region_2",
                "samples": [
                    {
                        "data": [8],
                        "modifiers": [
                            {
                                "data": [1],
                                "name": "staterror_region_2",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            },
        ],
        "measurements": [
            {"config": {"parameters": [], "poi": "Signal strength"}, "name": "My fit"}
        ],
        "observations": [
            {"data": [35, 8], "name": "region_1"},
            {"data": [10], "name": "region_2"},
        ],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_shapefactor():
    spec = {
        "channels": [
            {
                "name": "Signal Region",
                "samples": [
                    {
                        "data": [20, 10],
                        "modifiers": [
                            {
                                "data": None,
                                "name": "shape factor",
                                "type": "shapefactor",
                            },
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {"parameters": [], "poi": "Signal strength"},
                "name": "shapefactor fit",
            }
        ],
        "observations": [{"data": [25, 8], "name": "Signal Region"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_with_background():
    spec = {
        "channels": [
            {
                "name": "Signal Region",
                "samples": [
                    {
                        "data": [50],
                        "modifiers": [
                            {
                                "data": [5],
                                "name": "staterror_Signal-Region",
                                "type": "staterror",
                            },
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    },
                    {
                        "data": [150],
                        "modifiers": [
                            {
                                "data": [7],
                                "name": "staterror_Signal-Region",
                                "type": "staterror",
                            }
                        ],
                        "name": "Background",
                    },
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {"name": "Signal strength", "bounds": [[0, 10]], "inits": [1.0]}
                    ],
                    "poi": "Signal strength",
                },
                "name": "signal plus background",
            }
        ],
        "observations": [{"data": [160], "name": "Signal Region"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_no_aux():
    spec = {
        "channels": [
            {
                "name": "Signal Region",
                "samples": [
                    {
                        "data": [60],
                        "modifiers": [
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            }
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {"parameters": [], "poi": "Signal strength"},
                "name": "no auxdata",
            }
        ],
        "observations": [{"data": [65], "name": "Signal Region"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_lumi():
    spec = {
        "channels": [
            {
                "name": "Signal Region",
                "samples": [
                    {
                        "data": [35],
                        "modifiers": [
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                            {"data": None, "name": "lumi", "type": "lumi"},
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {
                            "auxdata": [1.0],
                            "inits": [1.05],
                            "name": "lumi",
                            "sigmas": [0.02],
                        }
                    ],
                    "poi": "Signal strength",
                },
                "name": "lumi modifier",
            }
        ],
        "observations": [{"data": [35], "name": "Signal Region"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_modifiers():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [35.0, 30.0],
                        "modifiers": [
                            {"data": None, "name": "mu", "type": "normfactor"},
                            {
                                "data": None,
                                "name": "mu_shapefactor",
                                "type": "shapefactor",
                            },
                            {
                                "data": {"hi": 0.8, "lo": 1.2},
                                "name": "normsys",
                                "type": "normsys",
                            },
                            {
                                "data": {
                                    "hi_data": [36.0, 31.0],
                                    "lo_data": [34.0, 29.0],
                                },
                                "name": "histosys",
                                "type": "histosys",
                            },
                            {
                                "data": [1.0, 2.0],
                                "name": "staterror_SR",
                                "type": "staterror",
                            },
                            {
                                "data": [1.0, 2.0],
                                "name": "shapesys_SR",
                                "type": "shapesys",
                            },
                            {"data": None, "name": "lumi", "type": "lumi"},
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {
                            "auxdata": [1.0],
                            "inits": [1.0],
                            "name": "lumi",
                            "sigmas": [0.02],
                        }
                    ],
                    "poi": "mu",
                },
                "name": "lumi modifier",
            }
        ],
        "observations": [{"data": [35, 30], "name": "SR"}],
        "version": "1.0.0",
    }
    return spec


@pytest.fixture
def example_spec_zero_staterror():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [5.0, 0.0],
                        "modifiers": [
                            {"data": None, "name": "mu", "type": "normfactor"},
                            {
                                "data": [1.0, 0.0],
                                "name": "staterror_SR",
                                "type": "staterror",
                            },
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [],
                    "poi": "mu",
                },
                "name": "zero staterror",
            }
        ],
        "observations": [{"data": [5, 0], "name": "SR"}],
        "version": "1.0.0",
    }
    return spec


# code below allows marking tests as slow and adds --runslow to run them
# implemented following https://docs.pytest.org/en/6.2.x/example/simple.html
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
