import numpy as np
import pyhf

from cabinetry import model_utils


def test_get_parameter_names(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = model_utils.get_parameter_names(model)
    assert labels == ["staterror_Signal-Region", "Signal strength"]


def test_get_asimov_parameters(example_spec):
    model = pyhf.Workspace(example_spec).model()
    pars, unc = model_utils.get_asimov_parameters(model)
    assert pars == [1.0, 1.0]
    assert np.allclose(unc, [0.0495665682, 0.0])

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

    model = pyhf.Workspace(spec).model()
    pars, unc = model_utils.get_asimov_parameters(model)
    assert pars == [1.0, 1.0, 1.0]
    assert unc == [0.0, 0.0, 0.0]
