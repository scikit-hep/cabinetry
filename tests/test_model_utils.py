import copy
import logging
from unittest import mock

import numpy as np
import pyhf

from cabinetry import model_utils
from cabinetry.fit.results_containers import FitResults


def test_ModelPrediction(example_spec):
    model = pyhf.Workspace(example_spec).model()
    model_yields = [[[10.0]]]
    total_stdev_model_bins = [[2.0]]
    total_stdev_model_channels = [2.0]
    label = "abc"
    model_prediction = model_utils.ModelPrediction(
        model, model_yields, total_stdev_model_bins, total_stdev_model_channels, label
    )
    assert model_prediction.model == model
    assert model_prediction.model_yields == model_yields
    assert model_prediction.total_stdev_model_bins == total_stdev_model_bins
    assert model_prediction.total_stdev_model_channels == total_stdev_model_channels
    assert model_prediction.label == label


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
    model, data = model_utils.model_and_data(example_spec, include_auxdata=False)
    assert data == [475]


def test_asimov_data(example_spec):
    ws = pyhf.Workspace(example_spec)
    model = ws.model()
    assert model_utils.asimov_data(model) == [51.8, 1]

    # without auxdata
    assert model_utils.asimov_data(model, include_auxdata=False) == [51.8]

    # respect nominal settings for normfactors
    example_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "inits": [2.0]}
    )
    ws = pyhf.Workspace(example_spec)
    model = ws.model()
    assert model_utils.asimov_data(model, include_auxdata=False) == [103.6]


def test_asimov_parameters(example_spec, example_spec_shapefactor, example_spec_lumi):
    model = pyhf.Workspace(example_spec).model()
    pars = model_utils.asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0])

    # respect shapefactor initial values
    example_spec_shapefactor["measurements"][0]["config"]["parameters"].append(
        {"name": "shape factor", "inits": [1.2, 1.1]}
    )
    model = pyhf.Workspace(example_spec_shapefactor).model()
    pars = model_utils.asimov_parameters(model)
    assert np.allclose(pars, [1.2, 1.1, 1.0])

    # respect normfactor initial values
    normfactor_spec = copy.deepcopy(example_spec)
    normfactor_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "inits": [2.0]}
    )
    model = pyhf.Workspace(normfactor_spec).model()
    pars = model_utils.asimov_parameters(model)
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
    pars = model_utils.asimov_parameters(model)
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
    pars = model_utils.asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0, 1.0])

    # lumi modifier with nominal value 1 and different initial value (ignored)
    model = pyhf.Workspace(example_spec_lumi).model()
    pars = model_utils.asimov_parameters(model)
    assert np.allclose(pars, [1.0, 1.0])


def test_prefit_uncertainties(
    example_spec, example_spec_multibin, example_spec_shapefactor
):
    model = pyhf.Workspace(example_spec).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.0])  # fixed parameter and normfactor

    model = pyhf.Workspace(example_spec_multibin).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.2, 0.4, 0.0, 0.125])

    model = pyhf.Workspace(example_spec_shapefactor).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.0, 0.0])


def test__channel_boundary_indices(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    indices = model_utils._channel_boundary_indices(model)
    assert indices == []

    model = pyhf.Workspace(example_spec_multibin).model()
    indices = model_utils._channel_boundary_indices(model)
    assert indices == [2]

    # add extra channel to model to test three channels (two indices needed)
    three_channel_model = copy.deepcopy(example_spec_multibin)
    extra_channel = copy.deepcopy(three_channel_model["channels"][0])
    extra_channel["name"] = "region_3"
    extra_channel["samples"][0]["modifiers"][0]["name"] = "staterror_region_3"
    three_channel_model["channels"].append(extra_channel)
    three_channel_model["observations"].append({"data": [35, 8], "name": "region_3"})
    model = pyhf.Workspace(three_channel_model).model()
    indices = model_utils._channel_boundary_indices(model)
    assert indices == [2, 3]


def test_yield_stdev(example_spec, example_spec_multibin):
    model = pyhf.Workspace(example_spec).model()
    parameters = np.asarray([1.05, 0.95])
    uncertainty = np.asarray([0.1, 0.1])
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])

    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model, parameters, uncertainty, corr_mat
    )
    assert np.allclose(total_stdev_bin, [[8.03150606]])
    assert np.allclose(total_stdev_chan, [8.03150606])

    # pre-fit
    parameters = np.asarray([1.0, 1.0])
    uncertainty = np.asarray([0.0495665682, 0.0])
    diag_corr_mat = np.diag([1.0, 1.0])
    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
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
    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model, parameters, uncertainty, corr_mat
    )
    expected_stdev_bin = [[8.056054, 1.670629], [2.775377]]
    expected_stdev_chan = [9.585327, 2.775377]
    for i_reg in range(2):
        assert np.allclose(total_stdev_bin[i_reg], expected_stdev_bin[i_reg])
        assert np.allclose(total_stdev_chan[i_reg], expected_stdev_chan[i_reg])

    # test caching by calling again with same arguments
    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model, parameters, uncertainty, corr_mat
    )
    for i_reg in range(2):
        assert np.allclose(total_stdev_bin[i_reg], expected_stdev_bin[i_reg])
        assert np.allclose(total_stdev_chan[i_reg], expected_stdev_chan[i_reg])
    # also look up cache directly
    from_cache = model_utils._YIELD_STDEV_CACHE[
        model, tuple(parameters), tuple(uncertainty), corr_mat.tobytes()
    ]
    for i_reg in range(2):
        assert np.allclose(from_cache[0][i_reg], expected_stdev_bin[i_reg])
        assert np.allclose(from_cache[1][i_reg], expected_stdev_chan[i_reg])


@mock.patch("cabinetry.model_utils.yield_stdev", return_value=([[0.3]], [0.3]))
@mock.patch(
    "cabinetry.model_utils.prefit_uncertainties", return_value=([0.04956657, 0.0])
)
@mock.patch("cabinetry.model_utils.asimov_parameters", return_value=([1.0, 1.0]))
def test_prediction(mock_asimov, mock_unc, mock_stdev, example_spec):
    model = pyhf.Workspace(example_spec).model()

    # pre-fit prediction
    model_pred = model_utils.prediction(model)
    assert model_pred.model == model
    assert model_pred.model_yields == [[[51.8]]]  # from pyhf expected_data call
    assert model_pred.total_stdev_model_bins == [[0.3]]  # from mock
    assert model_pred.total_stdev_model_channels == [0.3]  # from mock
    assert model_pred.label == "pre-fit"

    # Asimov parameter calculation and pre-fit uncertainties
    assert mock_asimov.call_args_list == [((model,), {})]
    assert mock_unc.call_args_list == [((model,), {})]

    # call to stdev calculation
    assert mock_stdev.call_count == 1
    assert mock_stdev.call_args_list[0][0][0] == model
    assert np.allclose(mock_stdev.call_args_list[0][0][1], [1.0, 1.0])
    assert np.allclose(mock_stdev.call_args_list[0][0][2], [0.04956657, 0.0])
    assert np.allclose(
        mock_stdev.call_args_list[0][0][3], np.asarray([[1.0, 0.0], [0.0, 1.0]])
    )
    assert mock_stdev.call_args_list[0][1] == {}

    # post-fit prediction
    fit_results = FitResults(
        np.asarray([1.01, 1.1]),
        np.asarray([0.03, 0.1]),
        [],
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        0.0,
    )
    model_pred = model_utils.prediction(model, fit_results=fit_results)
    assert model_pred.model == model
    assert np.allclose(model_pred.model_yields, [[[57.54980000]]])  # new par value
    assert model_pred.total_stdev_model_bins == [[0.3]]  # from mock
    assert model_pred.total_stdev_model_channels == [0.3]  # from mock
    assert model_pred.label == "post-fit"

    assert mock_asimov.call_count == 1  # no new call
    assert mock_unc.call_count == 1  # no new call

    # call to stdev calculation with fit_results propagated
    assert mock_stdev.call_count == 2
    assert mock_stdev.call_args_list[1][0][0] == model
    assert np.allclose(mock_stdev.call_args_list[1][0][1], [1.01, 1.1])
    assert np.allclose(mock_stdev.call_args_list[1][0][2], [0.03, 0.1])
    assert np.allclose(
        mock_stdev.call_args_list[1][0][3], np.asarray([[1.0, 0.2], [0.2, 1.0]])
    )
    assert mock_stdev.call_args_list[1][1] == {}

    # custom label
    model_pred = model_utils.prediction(model, label="abc")
    assert model_pred.label == "abc"


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


def test__parameter_index(caplog):
    caplog.set_level(logging.DEBUG)
    labels = ["a", "b", "c"]
    par_name = "b"
    assert model_utils._parameter_index(par_name, labels) == 1
    assert model_utils._parameter_index(par_name, tuple(labels)) == 1
    caplog.clear()

    assert model_utils._parameter_index("x", labels) == -1
    assert "parameter x not found in model" in [rec.message for rec in caplog.records]
    caplog.clear()


def test__strip_auxdata(example_spec):
    model = pyhf.Workspace(example_spec).model()
    data_with_aux = list(model.expected_data([1.0, 1.0], include_auxdata=True))
    data_without_aux = list(model.expected_data([1.0, 1.0], include_auxdata=False))

    assert model_utils._strip_auxdata(model, data_with_aux) == [51.8]
    assert model_utils._strip_auxdata(model, data_without_aux) == [51.8]


@mock.patch("cabinetry.model_utils._channel_boundary_indices", return_value=[2])
@mock.patch("cabinetry.model_utils._strip_auxdata", return_value=[25.0, 5.0, 8.0])
def test__data_per_channel(mock_aux, mock_bin, example_spec_multibin):
    model = pyhf.Workspace(example_spec_multibin).model()
    data = [25.0, 5.0, 8.0, 1.0, 1.0, 1.0]

    data_per_ch = model_utils._data_per_channel(model, [25.0, 5.0, 8.0, 1.0, 1.0, 1.0])
    assert data_per_ch == [[25.0, 5.0], [8.0]]  # auxdata stripped and split by channel

    # auxdata and channel index call
    assert mock_aux.call_args_list == [((model, data), {})]
    assert mock_bin.call_args_list == [((model,), {})]


def test__filter_channels(example_spec_multibin, caplog):
    caplog.set_level(logging.DEBUG)
    model = pyhf.Workspace(example_spec_multibin).model()

    assert model_utils._filter_channels(model, None) == ["region_1", "region_2"]
    assert model_utils._filter_channels(model, "region_1") == ["region_1"]
    assert model_utils._filter_channels(model, ["region_1", "region_2"]) == [
        "region_1",
        "region_2",
    ]
    assert model_utils._filter_channels(model, ["region_1", "abc"]) == ["region_1"]
    caplog.clear()

    # no matching channels
    assert model_utils._filter_channels(model, "abc") == []
    assert (
        "channel(s) ['abc'] not found in model, available channel(s): "
        "['region_1', 'region_2']" in [rec.message for rec in caplog.records]
    )
    caplog.clear()
