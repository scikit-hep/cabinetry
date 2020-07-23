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
        1.005119,
        0.981114,
        1.020708,
        0.982209,
        -0.213536,
        0.042937,
        0.857655,
    ]
    uncertainty_expected = [
        0.040951,
        0.037276,
        0.036499,
        0.042487,
        0.976741,
        0.160393,
        0.407588,
    ]
    best_twice_nll_expected = 16.274739734197926
    assert np.allclose(bestfit, bestfit_expected)
    assert np.allclose(uncertainty, uncertainty_expected)
    assert np.allclose(best_twice_nll, best_twice_nll_expected)
