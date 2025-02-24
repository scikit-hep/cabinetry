import copy
import logging
from unittest import mock

import numpy as np
import pyhf
import pytest

from cabinetry import model_utils
from cabinetry.fit.results_containers import FitResults


def test_LightModel(example_spec_with_multiple_background):
    # Test without merging samples
    model = pyhf.Workspace(example_spec_with_multiple_background).model()
    fit_results = FitResults(
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([0.1, 0.03, 0.07]),
        ["Background 2 norm", "Signal strength", "staterror_Signal-Region[0]"],
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        0.0,
    )
    model_pred = model_utils.prediction(
        model, fit_results=fit_results, samples_merge_map=None
    )

    assert model_pred.model.config.samples == ["Background", "Background 2", "Signal"]
    assert model_pred.model_yields == [[[150.0], [20.0], [50.0]]]
    assert model_pred.total_stdev_model_bins == [
        [[10.5], [2.441311123146742], [3.8078865529319543], [15.601602481796547]]
    ]
    assert model_pred.total_stdev_model_channels == [
        [10.5, 2.441311123146742, 3.8078865529319543, 15.601602481796547]
    ]
    assert model_pred.model.pyhf_model == model
    assert model_pred.model.config.channels == model.config.channels
    assert model_pred.model.config.channel_nbins == model.config.channel_nbins
    assert model_pred.model.config.channel_slices == model.config.channel_slices
    assert model_pred.model.config.modifier_settings == model.config.modifier_settings
    assert model_pred.model.spec == model.spec

    # Test with merging samples
    model_pred = model_utils.prediction(
        model,
        fit_results=fit_results,
        samples_merge_map={"Total Background": ["Background", "Background 2"]},
    )

    assert model_pred.model.config.samples == ["Total Background", "Signal"]
    assert model_pred.model_yields == [[[170.0], [50.0]]]
    assert model_pred.total_stdev_model_bins == [
        [[12.066896867049131], [3.8078865529319543], [15.601602481796547]]
    ]
    assert model_pred.total_stdev_model_channels == [
        [12.066896867049131, 3.8078865529319543, 15.601602481796547]
    ]
    assert model_pred.model.pyhf_model == model
    assert model_pred.model.config.channels == model.config.channels
    assert model_pred.model.config.channel_nbins == model.config.channel_nbins
    assert model_pred.model.config.channel_slices == model.config.channel_slices
    assert model_pred.model.config.modifier_settings == model.config.modifier_settings
    assert model_pred.model.spec == model.spec

    with pytest.raises(
        AttributeError,
        match="'LightConfig' object has no attribute 'NonExistent'",
    ):
        model_pred.model.config.NonExistent
    with pytest.raises(
        AttributeError,
        match="'LightModel' object has no attribute 'NonExistent'",
    ):
        model_pred.model.NonExistent


def test_ModelPrediction(example_spec):
    model = pyhf.Workspace(example_spec).model()
    model_yields = [[[10.0]]]
    total_stdev_model_bins = [[[2.0], [2.0]]]
    total_stdev_model_channels = [[2.0, 2.0]]
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
    model = pyhf.Workspace(example_spec).model()
    assert model_utils.asimov_data(model) == [51.8, 1]

    # without auxdata
    assert model_utils.asimov_data(model, include_auxdata=False) == [51.8]

    # respect nominal settings for normfactors
    example_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "inits": [2.0]}
    )
    model = pyhf.Workspace(example_spec).model()
    assert model_utils.asimov_data(model, include_auxdata=False) == [103.6]

    # post-fit Asimov
    fit_results = FitResults(
        np.asarray([3.0, 1.0]),
        np.asarray([0.1, 0.01]),
        ["Signal strength", "staterror_Signal-Region[0]"],
        np.asarray([[1.0, 0.1], [0.1, 1.0]]),
        0.0,
    )
    assert np.allclose(
        model_utils.asimov_data(model, fit_results=fit_results), [155.4, 1.0]
    )

    # post-fit Asimov, custom POI value (name from model config)
    assert np.allclose(
        model_utils.asimov_data(model, fit_results=fit_results, poi_value=1.5),
        [77.7, 1.0],
    )

    # pre-fit Asimov, custom POI name + value
    assert np.allclose(
        model_utils.asimov_data(
            model, poi_name="staterror_Signal-Region[0]", poi_value=1.1
        ),
        [113.96, 1.1],  # 2*51.8*1.1
    )

    # post-fit Asimov, no POI in model and no poi_name either
    example_spec["measurements"][0]["config"]["poi"] = ""
    model = pyhf.Workspace(example_spec).model()
    with pytest.raises(
        ValueError,
        match="no POI specified in model, use the poi_name argument to set POI name",
    ):
        model_utils.asimov_data(model, poi_value=1.0)


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
    assert np.allclose(pars, [1.0, 1.2, 1.1])

    # respect normfactor initial values
    normfactor_spec = copy.deepcopy(example_spec)
    normfactor_spec["measurements"][0]["config"]["parameters"].append(
        {"name": "Signal strength", "inits": [2.0]}
    )
    model = pyhf.Workspace(normfactor_spec).model()
    pars = model_utils.asimov_parameters(model)
    assert np.allclose(pars, [2.0, 1.0])

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
    assert np.allclose(pars, [1.0, 0.0, 1.0])

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
    example_spec,
    example_spec_multibin,
    example_spec_shapefactor,
    example_spec_zero_staterror,
):
    model = pyhf.Workspace(example_spec).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.0])  # fixed parameter and normfactor

    model = pyhf.Workspace(example_spec_multibin).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.2, 0.4, 0.125])

    model = pyhf.Workspace(example_spec_shapefactor).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.0, 0.0])

    model = pyhf.Workspace(example_spec_zero_staterror).model()
    unc = model_utils.prefit_uncertainties(model)
    assert np.allclose(unc, [0.0, 0.2, 0.0])  # partially fixed staterror


def test__hashable_model_key(example_spec):
    # key matches for two models built from the same spec
    model_1 = model_utils.LightModel(pyhf.Workspace(example_spec).model())
    model_2 = model_utils.LightModel(pyhf.Workspace(example_spec).model())
    assert model_utils._hashable_model_key(model_1) == model_utils._hashable_model_key(
        model_2
    )

    # key does not match if model has different interpcode
    model_new_interpcode = model_utils.LightModel(
        pyhf.Workspace(example_spec).model(
            modifier_settings={
                "normsys": {"interpcode": "code1"},
                "histosys": {"interpcode": "code0"},
            }
        )
    )

    assert model_utils._hashable_model_key(model_1) != model_utils._hashable_model_key(
        model_new_interpcode
    )


def test_yield_stdev(
    example_spec, example_spec_multibin, example_spec_with_multiple_background
):
    model = model_utils.LightModel(pyhf.Workspace(example_spec).model())
    parameters = np.asarray([0.95, 1.05])
    uncertainty = np.asarray([0.1, 0.1])
    corr_mat = np.asarray([[1.0, 0.2], [0.2, 1.0]])

    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model, parameters, uncertainty, corr_mat
    )
    assert np.allclose(total_stdev_bin, [[[8.03150606], [8.03150606]]])
    assert np.allclose(total_stdev_chan, [[8.03150606, 8.03150606]])

    # pre-fit
    parameters = np.asarray([1.0, 1.0])
    uncertainty = np.asarray([0.0, 0.0495665682])
    diag_corr_mat = np.diagflat([1.0, 1.0])
    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model, parameters, uncertainty, diag_corr_mat
    )
    assert np.allclose(total_stdev_bin, [[[2.56754823], [2.56754823]]])  # the staterror
    assert np.allclose(total_stdev_chan, [[2.56754823, 2.56754823]])

    # multiple channels, bins, staterrors
    model = model_utils.LightModel(pyhf.Workspace(example_spec_multibin).model())
    parameters = np.asarray([1.3, 0.9, 1.05, 0.95])
    uncertainty = np.asarray([0.3, 0.1, 0.05, 0.1])
    corr_mat = np.asarray(
        [
            [1.0, 0.2, 0.2, 0.3],
            [0.2, 1.0, 0.1, 0.1],
            [0.2, 0.1, 1.0, 0.3],
            [0.3, 0.1, 0.3, 1.0],
        ]
    )
    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model, parameters, uncertainty, corr_mat
    )
    expected_stdev_bin = [
        [[8.056054, 1.670629], [8.056054, 1.670629]],
        [[2.775377], [2.775377]],
    ]
    expected_stdev_chan = [[9.596340, 9.596340], [2.775377, 2.775377]]
    for i_reg in range(2):  # two channels
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
        model_utils._hashable_model_key(model),
        tuple(parameters),
        tuple(uncertainty),
        corr_mat.tobytes(),
        ",".join(model.config.samples),
    ]
    for i_reg in range(2):
        assert np.allclose(from_cache[0][i_reg], expected_stdev_bin[i_reg])
        assert np.allclose(from_cache[1][i_reg], expected_stdev_chan[i_reg])

    # Multiple backgrounds with sample merging
    # post-fit
    model = model_utils.LightModel(
        pyhf.Workspace(example_spec_with_multiple_background).model(),
        samples_merge_map={"Total Background": ["Background", "Background 2"]},
    )

    parameters = np.asarray([1.1, 1.01, 1.2])
    uncertainty = np.asarray([0.1, 0.03, 0.07])
    corr_mat = np.asarray([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])

    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model,
        parameters,
        uncertainty,
        corr_mat,
        samples_merge_map={"Total Background": ["Background", "Background 2"]},
    )
    assert np.allclose(
        total_stdev_bin,
        [[[12.510027977586642], [4.421993328805458], [16.66150128289767]]],
    )
    assert np.allclose(
        total_stdev_chan, [[12.510027977586642, 4.421993328805458, 16.66150128289767]]
    )

    # pre-fit
    parameters = np.asarray([1.0, 1.0, 1.0])
    uncertainty = np.asarray([0.1, 0.03, 0.07])
    diag_corr_mat = np.diagflat([1.0, 1.0, 1.0])
    total_stdev_bin, total_stdev_chan = model_utils.yield_stdev(
        model,
        parameters,
        uncertainty,
        diag_corr_mat,
        samples_merge_map={"Total Background": ["Background", "Background 2"]},
    )
    assert np.allclose(
        total_stdev_bin,
        [[[12.066896867049131], [3.8078865529319543], [15.601602481796547]]],
    )
    assert np.allclose(
        total_stdev_chan, [[12.066896867049131, 3.8078865529319543, 15.601602481796547]]
    )


@mock.patch(
    "cabinetry.model_utils.yield_stdev",
    side_effect=[
        (
            [[[5.0, 2.0], [5.0, 2.0]], [[1.0], [1.0]]],  # pre-fit, per-bin
            [[5.38516481, 5.38516481], [1.0, 1.0]],  # pre-fit, per-channel
        ),
        ([[[0.3], [0.3]]], [[0.3, 0.3]]),  # post-fit, single-channel nominal
        ([[[0.3], [0.3]]], [[0.3, 0.3]]),  # post-fit single-channel, custom
        (
            [[[12.5], [4.4], [16.7]]],
            [[12.5, 4.4, 16.7]],
        ),  # pre-fit single-channel, merged samples model
        (
            [[[12.5], [4.4], [16.7]]],
            [[12.5, 4.4, 16.7]],
        ),  # post-fit single-channel, merged samples model
    ],
)
@mock.patch(
    "cabinetry.model_utils.prefit_uncertainties",
    side_effect=[
        np.asarray([0.0, 0.2, 0.4, 0.125]),
        np.asarray([0.0, 0.0, 0.039]),  # pre-fit unc, merged samples model
    ],
)
@mock.patch(
    "cabinetry.model_utils.asimov_parameters",
    side_effect=[
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    ],
)
def test_prediction(
    mock_asimov,
    mock_unc,
    mock_stdev,
    example_spec_multibin,
    example_spec,
    example_spec_with_multiple_background,
    caplog,
):
    caplog.set_level(logging.DEBUG)
    model = pyhf.Workspace(example_spec_multibin).model()

    # pre-fit prediction, multi-channel model
    model_pred = model_utils.prediction(model)
    assert model_pred.model.pyhf_model == model
    # yields from pyhf expected_data call, per-bin / per-channel uncertainty from mock
    assert model_pred.model_yields == [[[25.0, 5.0]], [[8.0]]]
    assert model_pred.total_stdev_model_bins == [
        [[5.0, 2.0], [5.0, 2.0]],
        [[1.0], [1.0]],
    ]
    assert np.allclose(
        model_pred.total_stdev_model_channels, [[5.38516481, 5.38516481], [1.0, 1.0]]
    )
    assert model_pred.label == "pre-fit"

    # Asimov parameter calculation and pre-fit uncertainties
    assert mock_asimov.call_args_list[0] == ((model,), {})
    assert mock_unc.call_args_list[0] == ((model,), {})

    # call to stdev calculation
    assert mock_stdev.call_count == 1
    assert mock_stdev.call_args_list[0][0][0].pyhf_model == model
    assert np.allclose(mock_stdev.call_args_list[0][0][1], [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(mock_stdev.call_args_list[0][0][2], [0.0, 0.2, 0.4, 0.125])
    assert np.allclose(
        mock_stdev.call_args_list[0][0][3], np.diagflat([1.0, 1.0, 1.0, 1.0])
    )
    assert mock_stdev.call_args_list[0][1] == {"samples_merge_map": None}

    # post-fit prediction, single-channel model
    model = pyhf.Workspace(example_spec).model()
    fit_results = FitResults(
        np.asarray([1.1, 1.01]),
        np.asarray([0.1, 0.03]),
        ["Signal strength", "staterror_Signal-Region[0]"],
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        0.0,
    )
    model_pred = model_utils.prediction(model, fit_results=fit_results)
    assert model_pred.model.pyhf_model == model
    assert np.allclose(model_pred.model_yields, [[[57.54980000]]])  # new par value
    assert model_pred.total_stdev_model_bins == [[[0.3], [0.3]]]  # from mock
    assert model_pred.total_stdev_model_channels == [[0.3, 0.3]]  # from mock
    assert model_pred.label == "post-fit"
    assert "parameter names in fit results and model do not match" not in [
        rec.message for rec in caplog.records
    ]

    assert mock_asimov.call_count == 1  # no new call
    assert mock_unc.call_count == 1  # no new call

    # call to stdev calculation with fit_results propagated
    assert mock_stdev.call_count == 2
    assert mock_stdev.call_args_list[1][0][0].pyhf_model == model
    assert np.allclose(mock_stdev.call_args_list[1][0][1], [1.1, 1.01])
    assert np.allclose(mock_stdev.call_args_list[1][0][2], [0.1, 0.03])
    assert np.allclose(
        mock_stdev.call_args_list[1][0][3], np.asarray([[1.0, 0.2], [0.2, 1.0]])
    )
    assert mock_stdev.call_args_list[1][1] == {"samples_merge_map": None}

    caplog.clear()

    # custom prediction label, mismatch in parameter names
    fit_results = FitResults(
        np.asarray([1.1, 1.01]),
        np.asarray([0.1, 0.03]),
        ["a", "b"],
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        0.0,
    )
    model_pred = model_utils.prediction(model, fit_results=fit_results, label="abc")
    assert "parameter names in fit results and model do not match" in [
        rec.message for rec in caplog.records
    ]
    assert model_pred.label == "abc"
    caplog.clear()

    # Multiple backgrounds with sample merging
    model = pyhf.Workspace(example_spec_with_multiple_background).model()
    # pre-fit prediction, merged samples
    model_pred = model_utils.prediction(
        model, samples_merge_map={"Total Background": ["Background", "Background 2"]}
    )
    assert model_pred.model.pyhf_model == model
    # yields from pyhf expected_data call, per-bin / per-channel uncertainty from mock
    assert model_pred.model_yields == [[[170.0], [50.0]]]
    assert model_pred.total_stdev_model_bins == [[[12.5], [4.4], [16.7]]]
    assert model_pred.total_stdev_model_channels == [[12.5, 4.4, 16.7]]
    assert model_pred.label == "pre-fit"

    # Asimov parameter calculation and pre-fit uncertainties
    assert mock_asimov.call_args_list[1] == ((model,), {})
    assert mock_unc.call_args_list[1] == ((model,), {})

    assert mock_asimov.call_count == 2  # one new call
    assert mock_unc.call_count == 2  # one new call

    # call to stdev calculation
    assert mock_stdev.call_count == 4
    assert mock_stdev.call_args_list[3][1] == {
        "samples_merge_map": {"Total Background": ["Background", "Background 2"]}
    }

    # post-fit
    fit_results = FitResults(
        np.asarray([1.2, 1.1, 1.01]),
        np.asarray([0.1, 0.03, 0.07]),
        ["Background 2 norm", "Signal strength", "staterror_Signal-Region[0]"],
        np.asarray([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]),
        0.0,
    )
    model_pred = model_utils.prediction(
        model,
        fit_results=fit_results,
        samples_merge_map={"Total Background": ["Background", "Background 2"]},
    )
    assert model_pred.model.pyhf_model == model
    assert np.allclose(model_pred.model_yields, [[[175.74], [55.55]]])  # new par value
    assert model_pred.total_stdev_model_bins == [[[12.5], [4.4], [16.7]]]  # from mock
    assert model_pred.total_stdev_model_channels == [[12.5, 4.4, 16.7]]  # from mock
    assert model_pred.label == "post-fit"
    assert "parameter names in fit results and model do not match" not in [
        rec.message for rec in caplog.records
    ]

    # Asimov parameter calculation and pre-fit uncertainties
    assert mock_asimov.call_count == 2  # no new call
    assert mock_unc.call_count == 2  # no new call

    # call to stdev calculation
    assert mock_stdev.call_count == 5
    assert mock_stdev.call_args_list[4][1] == {
        "samples_merge_map": {"Total Background": ["Background", "Background 2"]}
    }

    caplog.clear()


def test_unconstrained_parameter_count(example_spec, example_spec_shapefactor):
    model = pyhf.Workspace(example_spec).model()
    assert model_utils.unconstrained_parameter_count(model) == 1

    model = pyhf.Workspace(example_spec_shapefactor).model()
    assert model_utils.unconstrained_parameter_count(model) == 3

    # fixed parameters can be provided to function
    model = pyhf.Workspace(example_spec_shapefactor).model()
    fix_pars = model.config.suggested_fixed()
    fix_pars[0] = True
    assert model_utils.unconstrained_parameter_count(model, fix_pars=fix_pars) == 2

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

    assert model_utils._parameter_index("x", labels) is None
    assert "parameter x not found in model" in [rec.message for rec in caplog.records]
    caplog.clear()


def test__poi_index(example_spec, caplog):
    caplog.set_level(logging.DEBUG)
    model = pyhf.Workspace(example_spec).model()

    # POI given by name
    assert model_utils._poi_index(model, poi_name="staterror_Signal-Region[0]") == 1

    # parameter not found in model
    with pytest.raises(ValueError, match="parameter x not found in model"):
        model_utils._poi_index(model, poi_name="x")

    # POI specified in model config
    assert model_utils._poi_index(model) == 0
    caplog.clear()

    # no POI
    example_spec["measurements"][0]["config"]["poi"] = ""
    model = pyhf.Workspace(example_spec).model()
    assert model_utils._poi_index(model) is None
    assert "could not find POI for model" in [rec.message for rec in caplog.records]
    caplog.clear()


def test__strip_auxdata(example_spec):
    model = pyhf.Workspace(example_spec).model()
    light_model = model_utils.LightModel(model)
    data_with_aux = list(model.expected_data([1.0, 1.0], include_auxdata=True))
    data_without_aux = list(model.expected_data([1.0, 1.0], include_auxdata=False))

    assert model_utils._strip_auxdata(light_model, data_with_aux) == [51.8]
    assert model_utils._strip_auxdata(light_model, data_without_aux) == [51.8]


@mock.patch("cabinetry.model_utils._strip_auxdata", return_value=[25.0, 5.0, 8.0])
def test__data_per_channel(mock_aux, example_spec_multibin):
    model = pyhf.Workspace(example_spec_multibin).model()
    light_model = model_utils.LightModel(model)
    data = [25.0, 5.0, 8.0, 1.0, 1.0, 1.0]

    data_per_ch = model_utils._data_per_channel(
        light_model, [25.0, 5.0, 8.0, 1.0, 1.0, 1.0]
    )
    assert data_per_ch == [[25.0, 5.0], [8.0]]  # auxdata stripped and split by channel

    # auxdata and channel index call
    assert mock_aux.call_args_list == [((light_model, data), {})]


def test__filter_channels(example_spec_multibin, caplog):
    caplog.set_level(logging.DEBUG)
    model = pyhf.Workspace(example_spec_multibin).model()
    light_model = model_utils.LightModel(model)
    assert model_utils._filter_channels(light_model, None) == ["region_1", "region_2"]
    assert model_utils._filter_channels(light_model, "region_1") == ["region_1"]
    assert model_utils._filter_channels(light_model, ["region_1", "region_2"]) == [
        "region_1",
        "region_2",
    ]
    assert model_utils._filter_channels(light_model, ["region_1", "abc"]) == [
        "region_1"
    ]
    caplog.clear()

    # no matching channels
    assert model_utils._filter_channels(light_model, "abc") == []
    assert (
        "channel(s) ['abc'] not found in model, available channel(s): "
        "['region_1', 'region_2']" in [rec.message for rec in caplog.records]
    )
    caplog.clear()


@mock.patch(
    "cabinetry.model_utils.prefit_uncertainties",
    side_effect=[[0.4, 0.5, 0.6], [0.4, 0.5], [0.4, 0.5, 0.6]],
)
@mock.patch(
    "cabinetry.model_utils.asimov_parameters",
    side_effect=[[4.0, 5.0, 6.0], [4.0, 5.0], [4.0, 5.0, 6.0]],
)
def test_match_fit_results(mock_pars, mock_uncs):
    mock_model = mock.MagicMock()
    fit_results = FitResults(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.1, 0.2, 0.3]),
        ["par_a", "par_b", "par_c"],
        np.asarray([[1.0, 0.2, 0.5], [0.2, 1.0, 0.1], [0.5, 0.1, 1.0]]),
        5.0,
        0.1,
    )

    # remove par_a, flip par_b and par_c, add par_d
    mock_model.config.par_names = ["par_c", "par_d", "par_b"]
    matched_fit_res = model_utils.match_fit_results(mock_model, fit_results)
    assert mock_pars.call_args_list == [((mock_model,), {})]
    assert mock_uncs.call_args_list == [((mock_model,), {})]
    assert np.allclose(matched_fit_res.bestfit, [3.0, 5.0, 2.0])
    assert np.allclose(matched_fit_res.uncertainty, [0.3, 0.5, 0.2])
    assert matched_fit_res.labels == ["par_c", "par_d", "par_b"]
    assert np.allclose(
        matched_fit_res.corr_mat, [[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]]
    )
    assert matched_fit_res.best_twice_nll == 5.0
    assert matched_fit_res.goodness_of_fit == 0.1

    # all parameters are new
    mock_model.config.par_names = ["par_d", "par_e"]
    matched_fit_res = model_utils.match_fit_results(mock_model, fit_results)
    assert np.allclose(matched_fit_res.bestfit, [4.0, 5.0])
    assert np.allclose(matched_fit_res.uncertainty, [0.4, 0.5])
    assert matched_fit_res.labels == ["par_d", "par_e"]
    assert np.allclose(matched_fit_res.corr_mat, [[1.0, 0.0], [0.0, 1.0]])
    assert matched_fit_res.best_twice_nll == 5.0
    assert matched_fit_res.goodness_of_fit == 0.1

    # fit results already match model exactly
    mock_model.config.par_names = ["par_a", "par_b", "par_c"]
    matched_fit_res = model_utils.match_fit_results(mock_model, fit_results)
    assert np.allclose(matched_fit_res.bestfit, [1.0, 2.0, 3.0])
    assert np.allclose(matched_fit_res.uncertainty, [0.1, 0.2, 0.3])
    assert matched_fit_res.labels == ["par_a", "par_b", "par_c"]
    assert np.allclose(
        matched_fit_res.corr_mat, [[1.0, 0.2, 0.5], [0.2, 1.0, 0.1], [0.5, 0.1, 1.0]]
    )
    assert matched_fit_res.best_twice_nll == 5.0
    assert matched_fit_res.goodness_of_fit == 0.1


def test_modifier_map(example_spec):
    model = pyhf.Workspace(example_spec).model()

    modifier_map = model_utils._modifier_map(model)
    assert modifier_map == {
        ("Signal Region", "Signal", "staterror_Signal-Region"): ["staterror"],
        ("Signal Region", "Signal", "Signal strength"): ["normfactor"],
    }
    assert modifier_map[("a", "b", "c")] == []


def test__parameters_maximizing_constraint_term(
    example_spec, example_spec_no_aux, example_spec_modifiers
):
    # staterror, custom auxdata value
    model = pyhf.Workspace(example_spec).model()
    best_pars = model_utils._parameters_maximizing_constraint_term(model, [1.1])
    assert np.allclose(best_pars, [1.0, 1.1])

    # no auxdata
    model = pyhf.Workspace(example_spec_no_aux).model()
    best_pars = model_utils._parameters_maximizing_constraint_term(model, [])
    assert np.allclose(best_pars, [1.0])

    # all modifiers (including Poisson constraint for shapesys), custom auxdata values
    # order: histosys, lumi, normfactor, normsys, shapefactor (2 bins),
    # shapesys (2 bins), staterror (2 bins)
    model = pyhf.Workspace(example_spec_modifiers).model()
    best_pars = model_utils._parameters_maximizing_constraint_term(
        model, [0.1, 1.1, 0.2, 1531.25, 281.25, 0.9, 0.8]
    )
    assert np.allclose(best_pars, [0.1, 1.1, 1.0, 0.2, 1.0, 1.0, 1.25, 1.25, 0.9, 0.8])
