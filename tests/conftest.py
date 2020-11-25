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
                        "data": [51.839756],
                        "modifiers": [
                            {
                                "data": [2.5695188],
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
