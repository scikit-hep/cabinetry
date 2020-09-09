import awkward1 as ak
import numpy as np
import pyhf

from cabinetry import model_utils


def test_model_and_data(example_spec):
    model, data = model_utils.model_and_data(example_spec)
    assert model.spec["channels"] == example_spec["channels"]
    assert model.config.modifier_settings == {
        "normsys": {"interpcode": "code4"},
        "histosys": {"interpcode": "code4p"},
    }
    assert data == [475, 1.0]

    # requesting Asimov dataset
    model, data = model_utils.model_and_data(example_spec, asimov=True)
    assert data == [51.839756, 1.0]

    # without auxdata
    model, data = model_utils.model_and_data(example_spec, with_aux=False)
    assert data == [475]


def test_get_parameter_names(example_spec):
    model = pyhf.Workspace(example_spec).model()
    labels = model_utils.get_parameter_names(model)
    assert labels == ["staterror_Signal-Region", "Signal strength"]


def test_build_Asimov_data(example_spec):
    ws = pyhf.Workspace(example_spec)
    model = ws.model()
    assert model_utils.build_Asimov_data(model) == [51.839756, 1]

    # without auxdata
    assert model_utils.build_Asimov_data(model, with_aux=False) == [51.839756]


def test_get_asimov_parameters(example_spec):
    model = pyhf.Workspace(example_spec).model()
    pars, unc = model_utils.get_asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0])
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
    assert np.allclose(pars, [1.0, 1.0, 1.0])
    assert np.allclose(unc, [0.0, 0.0, 0.0])


def test_calculate_stdev(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    parameters = np.asarray([1.05, 0.95])
    uncertainty = np.asarray([0.1, 0.1])
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])

    total_stdev = model_utils.calculate_stdev(model, parameters, uncertainty, corr_mat)
    expected_stdev = [[8.03767016]]
    assert np.allclose(ak.to_list(total_stdev), expected_stdev)

    # pre-fit
    parameters = np.asarray([1.0, 1.0])
    uncertainty = np.asarray([0.0495665682, 0.0])
    diag_corr_mat = np.diag([1.0, 1.0])
    total_stdev = model_utils.calculate_stdev(
        model, parameters, uncertainty, diag_corr_mat
    )
    expected_stdev = [[2.56951880]]  # the staterror
    assert np.allclose(ak.to_list(total_stdev), expected_stdev)

    # multiple channels, bins, staterrors
    model = pyhf.Workspace(example_spec_multibin).model()
    parameters = np.asarray([0.9, 1.05, 1.3, 0.95])
    uncertainty = np.asarray([0.1, 0.05, 0.3, 0.1])
    corr_mat = np.asarray(
        [
            [1.0, 0.1, 0.2, 0.1],
            [0.1, 1.0, 0.2, 0.3],
            [0.2, 0.2, 1.0, 0.3],
            [0.1, 0.3, 0.3, 1.0],
        ]
    )
    total_stdev = model_utils.calculate_stdev(model, parameters, uncertainty, corr_mat)
    expected_stdev = [[8.056054, 1.670629], [2.775377]]
    for i_reg in range(2):
        assert np.allclose(ak.to_list(total_stdev[i_reg]), expected_stdev[i_reg])
