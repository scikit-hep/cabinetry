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
    histo_folder = str(tmp_path) + "/histograms/"
    cabinetry.template_builder.create_histograms(
        cabinetry_config, histo_folder, method="uproot"
    )
    cabinetry.template_postprocessor.run(cabinetry_config, histo_folder)
    workspace_path = "workspaces/example_workspace.json"
    ws = cabinetry.workspace.build(cabinetry_config, histo_folder)
    cabinetry.workspace.save(ws, workspace_path)
    ws = cabinetry.workspace.load(workspace_path)
    bestfit, uncertainty, _, best_twice_nll = cabinetry.fit.fit(ws)

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
    assert np.allclose(bestfit, bestfit_expected)
    assert np.allclose(uncertainty, uncertainty_expected)
    assert np.allclose(best_twice_nll, best_twice_nll_expected)
