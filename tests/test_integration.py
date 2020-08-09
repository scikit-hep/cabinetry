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
        1.00124934,
        0.98903044,
        1.01966220,
        0.98309447,
        -0.08539741,
        -0.36008148,
        -0.59091951,
        1.71110973,
    ]
    uncertainty_expected = [
        0.04112787,
        0.03806766,
        0.03650754,
        0.04249824,
        0.98724460,
        0.48031552,
        0.62231780,
        0.90353740,
    ]
    best_twice_nll_expected = 17.199087
    corr_mat_expected = [
        [
            1.0,
            0.05741159,
            0.01948892,
            -0.02326599,
            0.0077079,
            0.19230206,
            0.08525489,
            -0.12724928,
        ],
        [
            0.05741159,
            1.0,
            0.00979333,
            -0.00597878,
            -0.01864312,
            -0.12725387,
            -0.20681375,
            0.16167847,
        ],
        [
            0.01948892,
            0.00979333,
            1.0,
            0.06583417,
            0.00240035,
            0.04198489,
            0.02614081,
            -0.11107066,
        ],
        [
            -0.02326599,
            -0.00597878,
            0.06583417,
            1.0,
            -0.00156359,
            -0.03826249,
            -0.0178134,
            -0.06751349,
        ],
        [
            0.0077079,
            -0.01864312,
            0.00240035,
            -0.00156359,
            1.0,
            0.08306285,
            -0.14175595,
            -0.17519639,
        ],
        [
            0.19230206,
            -0.12725387,
            0.04198489,
            -0.03826249,
            0.08306285,
            1.0,
            0.92056977,
            -0.92448677,
        ],
        [
            0.08525489,
            -0.20681375,
            0.02614081,
            -0.0178134,
            -0.14175595,
            0.92056977,
            1.0,
            -0.88499252,
        ],
        [
            -0.12724928,
            0.16167847,
            -0.11107066,
            -0.06751349,
            -0.17519639,
            -0.92448677,
            -0.88499252,
            1.0,
        ],
    ]
    assert np.allclose(bestfit, bestfit_expected)
    assert np.allclose(uncertainty, uncertainty_expected)
    assert np.allclose(best_twice_nll, best_twice_nll_expected)
    assert np.allclose(corr_mat, corr_mat_expected, rtol=5e-5)
