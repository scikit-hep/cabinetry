import logging

import numpy as np
import pyhf
import pytest

from cabinetry import fit


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


def test_get_parameter_names(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = fit.get_parameter_names(model)
    assert labels == ["staterror_Signal-Region", "Signal strength"]


def test_print_results(caplog):
    caplog.set_level(logging.DEBUG)

    bestfit = np.asarray([1.0, 2.0])
    uncertainty = np.asarray([0.1, 0.3])
    labels = ["param_A", "param_B"]
    fit.print_results(bestfit, uncertainty, labels)
    assert "param_A:  1.000000 +/- 0.100000" in [rec.message for rec in caplog.records]
    assert "param_B:  2.000000 +/- 0.300000" in [rec.message for rec in caplog.records]
    caplog.clear()


def test_build_Asimov_data(example_spec):
    ws = pyhf.Workspace(example_spec)
    model = ws.model()
    assert np.allclose(fit.build_Asimov_data(model), [51.839756, 1])


# skip a "RuntimeWarning: numpy.ufunc size changed" warning
# due to different numpy versions used in dependencies
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fit(example_spec):
    bestfit, uncertainty, labels, best_twice_nll, corr_mat = fit.fit(example_spec)
    assert np.allclose(bestfit, [1.00016745, 9.16046294])
    assert np.allclose(uncertainty, [0.04953864, 0.6131055])
    assert labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(best_twice_nll, 3.83055796)
    assert np.allclose(corr_mat, [[1.0, -0.73346895], [-0.73346895, 1.0]])

    # Asimov fit
    bestfit, uncertainty, labels, best_twice_nll, corr_mat = fit.fit(
        example_spec, asimov=True
    )
    assert np.allclose(bestfit, [1.0, 1.0], rtol=1e-4)
    assert np.allclose(uncertainty, [0.04956413, 0.1474005])
    assert labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(best_twice_nll, 1.61824936)
    assert np.allclose(corr_mat, [[1.0, -0.33610697], [-0.33610697, 1.0]])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_custom_fit(example_spec):
    bestfit, uncertainty, labels, best_twice_nll, corr_mat = fit.custom_fit(
        example_spec
    )
    # compared to fit(), the gamma is fixed
    assert np.allclose(bestfit, [1.1, 8.32985794])
    assert np.allclose(uncertainty, [0.0, 0.38153392])
    assert labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(best_twice_nll, 7.90080378)
    assert np.allclose(corr_mat, [[0.0, 0.0], [0.0, 1.0]])

    # Asimov fit, with fixed gamma (fixed not to Asimov MLE)
    bestfit, uncertainty, labels, best_twice_nll, corr_mat = fit.custom_fit(
        example_spec, asimov=True
    )
    # the gamma factor is multiplicative and fixed to 1.1, so the
    # signal strength needs to be 1/1.1 to compensate
    assert np.allclose(bestfit, [1.1, 0.90917877])
    assert np.allclose(uncertainty, [0.0, 0.12623172])
    assert labels == ["staterror_Signal-Region", "Signal strength"]
    assert np.allclose(best_twice_nll, 5.68851093)
    assert np.allclose(corr_mat, [[0.0, 0.0], [0.0, 1.0]])
