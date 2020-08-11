import numpy as np
import pytest

import cabinetry
from util import create_ntuples


@pytest.fixture
def ntuple_creator():
    return create_ntuples.run


@pytest.mark.no_cover
def test_integration(tmp_path, ntuple_creator):
    """The purpose of this integration test is to check whether the
    steps run without error and whether the fit result is as expected.
    """
    ntuple_creator(str(tmp_path) + "/")

    cabinetry_config = cabinetry.configuration.read("config_example.yml")
    cabinetry.template_builder.create_histograms(cabinetry_config, method="uproot")
    cabinetry.template_postprocessor.run(cabinetry_config)
    workspace_path = "workspaces/example_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config)
    cabinetry.workspace.save(ws, workspace_path)
    ws = cabinetry.workspace.load(workspace_path)
    bestfit, uncertainty, _, best_twice_nll, corr_mat = cabinetry.fit.fit(ws)

    bestfit_expected = [
        1.00097604,
        0.98918703,
        1.01954935,
        0.98300027,
        -0.09142767,
        -0.32313458,
        -0.58374164,
        1.68652611,
    ]
    uncertainty_expected = [
        0.0410808,
        0.03792673,
        0.03651597,
        0.04250513,
        0.99112844,
        0.55537361,
        0.627469,
        0.92411694,
    ]
    best_twice_nll_expected = 17.194286
    corr_mat_expected = [
        [
            1.0,
            0.0628096592,
            0.0195177192,
            -0.0235685409,
            0.0106844629,
            0.162391048,
            0.0674859362,
            -0.111109465,
        ],
        [
            0.0628096592,
            1.0,
            0.00904907811,
            -0.0049250707,
            -0.0213145107,
            -0.122847225,
            -0.189009516,
            0.147074209,
        ],
        [
            0.0195177192,
            0.00904907811,
            1.0,
            0.0654356196,
            0.00166640856,
            0.0463716356,
            0.0327822936,
            -0.11511303,
        ],
        [
            -0.0235685409,
            -0.0049250707,
            0.0654356196,
            1.0,
            -0.000760862587,
            -0.0428518642,
            -0.0253515109,
            -0.0586033381,
        ],
        [
            0.0106844629,
            -0.0213145107,
            0.00166640856,
            -0.000760862587,
            1.0,
            0.0530075596,
            -0.137965149,
            -0.167610909,
        ],
        [
            0.162391048,
            -0.122847225,
            0.0463716356,
            -0.0428518642,
            0.0530075596,
            1.0,
            0.941499622,
            -0.93124935,
        ],
        [
            0.0674859362,
            -0.189009516,
            0.0327822936,
            -0.0253515109,
            -0.137965149,
            0.941499622,
            1.0,
            -0.891613116,
        ],
        [
            -0.111109465,
            0.147074209,
            -0.11511303,
            -0.0586033381,
            -0.167610909,
            -0.93124935,
            -0.891613116,
            1.0,
        ],
    ]
    assert np.allclose(bestfit, bestfit_expected)
    assert np.allclose(uncertainty, uncertainty_expected)
    assert np.allclose(best_twice_nll, best_twice_nll_expected)
    assert np.allclose(corr_mat, corr_mat_expected, rtol=5e-5)
