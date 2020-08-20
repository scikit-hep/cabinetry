import pytest
import uproot


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
