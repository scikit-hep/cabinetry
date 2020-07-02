import logging

import numpy as np
import pyhf
import pytest

from cabinetry import fit


@pytest.fixture
def example_spec():
    # fmt: off
    spec = {"channels": [{"name": "Signal Region", "samples": [{"data": [51.839756],
            "modifiers":[{"data": [2.5695188], "name": "staterror_Signal-Region",
            "type": "staterror"}, {"data": None, "name": "Signal strength", "type": "normfactor"}],
            "name": "Signal"}]}], "measurements": [{"config": {"parameters": [],
            "poi": "Signal strength"}, "name":"My fit"}], "observations":
            [{"data": [475], "name": "Signal Region"}], "version": "1.0.0"}
    # fmt: on
    return spec


def test_get_parameter_names(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = fit.get_parameter_names(model)
    assert labels == ["staterror_Signal-Region", "Signal strength"]


def test_print_results(caplog):
    caplog.set_level(logging.DEBUG)

    bestfit = [1.0, 2.0]
    uncertainty = [0.1, 0.3]
    labels = ["param_A", "param_B"]
    fit.print_results(bestfit, uncertainty, labels)
    assert "param_A: 1.000000 +/- 0.100000" in [rec.message for rec in caplog.records]
    assert "param_B: 2.000000 +/- 0.300000" in [rec.message for rec in caplog.records]
    caplog.clear()


# skip a "RuntimeWarning: numpy.ufunc size changed" warning
# due to different numpy versions used in dependencies
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fit(example_spec):
    bestfit, uncertainty, labels = fit.fit(example_spec)
    assert np.allclose(bestfit, [0.99998772, 9.16255687])
    assert np.allclose(uncertainty, [0.04954955, 0.61348804])
    assert labels == ["staterror_Signal-Region", "Signal strength"]
