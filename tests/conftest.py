import pytest
import uproot3 as uproot


class Utils:
    @staticmethod
    def create_ntuple(fname, treename, varname, var_array, weightname, weight_array):
        with uproot.recreate(fname) as f:
            f[treename] = uproot.newtree({varname: "float64", weightname: "float64"})
            f[treename].extend({varname: var_array, weightname: weight_array})


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
                        {
                            "name": "Signal strength",
                            "bounds": [[0, 10]],
                            "inits": [1.0],
                        }
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
                "name": "no auxdata",
            }
        ],
        "observations": [{"data": [65], "name": "Signal Region"}],
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
