import copy
import logging

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
    assert data == [51.8, 1.0]

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
    assert model_utils.build_Asimov_data(model) == [51.8, 1]

    # without auxdata
    assert model_utils.build_Asimov_data(model, with_aux=False) == [51.8]

    # respect nominal settings for normfactors
    example_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "inits": [2.0]}
    )
    ws = pyhf.Workspace(example_spec)
    model = ws.model()
    assert model_utils.build_Asimov_data(model, with_aux=False) == [103.6]


def test_get_asimov_parameters(example_spec, example_spec_shapefactor):
    model = pyhf.Workspace(example_spec).model()
    pars = model_utils.get_asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0])

    # respect shapefactor initial values
    example_spec_shapefactor["measurements"][0]["config"]["parameters"].append(
        {"name": "shape factor", "inits": [1.2, 1.1]}
    )
    model = pyhf.Workspace(example_spec_shapefactor).model()
    pars = model_utils.get_asimov_parameters(model)
    assert np.allclose(pars, [1.2, 1.1, 1.0])

    # respect normfactor initial values
    normfactor_spec = copy.deepcopy(example_spec)
    normfactor_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "inits": [2.0]}
    )
    model = pyhf.Workspace(normfactor_spec).model()
    pars = model_utils.get_asimov_parameters(model)
    assert np.allclose(pars, [1.0, 2.0])

    # modifier with nominal value 0 and different initial value (which is ignored)
    normsys_spec = copy.deepcopy(example_spec)
    normsys_spec["channels"][0]["samples"][0]["modifiers"].append(
        {"data": {"hi": 1.2, "lo": 0.8}, "name": "normsys_example", "type": "normsys"}
    )
    normsys_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "normsys_example", "inits": [0.5]}
    )
    model = pyhf.Workspace(normsys_spec).model()
    pars = model_utils.get_asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0, 0.0])

    # shapesys modifier with nominal value 1 and different initial value (ignored)
    shapesys_spec = copy.deepcopy(example_spec)
    shapesys_spec["channels"][0]["samples"][0]["modifiers"].append(
        {"data": [5.0], "name": "shapesys_example", "type": "shapesys"}
    )
    shapesys_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "shapesys_example", "inits": [1.5]}
    )
    model = pyhf.Workspace(shapesys_spec).model()
    pars = model_utils.get_asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0, 1.0])


def test_get_prefit_uncertainties(
    example_spec, example_spec_multibin, example_spec_shapefactor
):
    model = pyhf.Workspace(example_spec).model()
    unc = model_utils.get_prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.0])  # fixed parameter and normfactor

    model = pyhf.Workspace(example_spec_multibin).model()
    unc = model_utils.get_prefit_uncertainties(model)
    assert np.allclose(unc, [0.2, 0.4, 0.0, 0.125])

    model = pyhf.Workspace(example_spec_shapefactor).model()
    unc = model_utils.get_prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.0, 0.0])


def test__get_channel_boundary_indices(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    indices = model_utils._get_channel_boundary_indices(model)
    assert indices == []

    model = pyhf.Workspace(example_spec_multibin).model()
    indices = model_utils._get_channel_boundary_indices(model)
    assert indices == [2]

    # add extra channel to model to test three channels (two indices needed)
    three_channel_model = copy.deepcopy(example_spec_multibin)
    extra_channel = copy.deepcopy(three_channel_model["channels"][0])
    extra_channel["name"] = "region_3"
    extra_channel["samples"][0]["modifiers"][0]["name"] = "staterror_region_3"
    three_channel_model["channels"].append(extra_channel)
    three_channel_model["observations"].append({"data": [35, 8], "name": "region_3"})
    model = pyhf.Workspace(three_channel_model).model()
    indices = model_utils._get_channel_boundary_indices(model)
    assert indices == [2, 3]


def test_calculate_stdev(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    parameters = np.asarray([1.05, 0.95])
    uncertainty = np.asarray([0.1, 0.1])
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])

    total_stdev_bin, total_stdev_chan = model_utils.calculate_stdev(
        model, parameters, uncertainty, corr_mat
    )
    assert np.allclose(total_stdev_bin, [[8.03150606]])
    assert np.allclose(total_stdev_chan, [8.03150606])

    # pre-fit
    parameters = np.asarray([1.0, 1.0])
    uncertainty = np.asarray([0.0495665682, 0.0])
    diag_corr_mat = np.diag([1.0, 1.0])
    total_stdev_bin, total_stdev_chan = model_utils.calculate_stdev(
        model, parameters, uncertainty, diag_corr_mat
    )
    assert np.allclose(total_stdev_bin, [[2.56754823]])  # the staterror
    assert np.allclose(total_stdev_chan, [2.56754823])

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
    total_stdev_bin, total_stdev_chan = model_utils.calculate_stdev(
        model, parameters, uncertainty, corr_mat
    )
    expected_stdev_bin = [[8.056054, 1.670629], [2.775377]]
    expected_stdev_chan = [9.585327, 2.775377]
    for i_reg in range(2):
        assert np.allclose(total_stdev_bin[i_reg], expected_stdev_bin[i_reg])
        assert np.allclose(total_stdev_chan[i_reg], expected_stdev_chan[i_reg])


def test_unconstrained_parameter_count(example_spec, example_spec_shapefactor):
    model = pyhf.Workspace(example_spec).model()
    assert model_utils.unconstrained_parameter_count(model) == 1

    model = pyhf.Workspace(example_spec_shapefactor).model()
    assert model_utils.unconstrained_parameter_count(model) == 3

    # fixed parameters are skipped in counting
    example_spec_shapefactor["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "fixed": True}
    )
    model = pyhf.Workspace(example_spec_shapefactor).model()
    assert model_utils.unconstrained_parameter_count(model) == 2


def test__get_parameter_index(caplog):
    caplog.set_level(logging.DEBUG)
    labels = ["a", "b", "c"]
    par_name = "b"
    assert model_utils._get_parameter_index(par_name, labels) == 1
    assert model_utils._get_parameter_index(par_name, tuple(labels)) == 1
    caplog.clear()

    assert model_utils._get_parameter_index("x", labels) == -1
    assert "parameter x not found in model" in [rec.message for rec in caplog.records]
    caplog.clear()
