from unittest import mock

import numpy as np
import pytest

from cabinetry import fit
from cabinetry import model_utils
from cabinetry.fit import utils


@mock.patch(
    "cabinetry.fit._fit_model",
    side_effect=[
        # ranking call with reference fit results (auxdata shift, 2 calls)
        fit.FitResults(
            np.asarray([0.15] * 10),
            np.asarray([0.01] * 10),
            np.full((10, 10), 0.1),
            np.empty(0),
            0.0,
        ),
        fit.FitResults(
            np.asarray([0.05] * 10),
            np.asarray([0.01] * 10),
            np.full((10, 10), 0.1),
            np.empty(0),
            0.0,
        ),
    ],
)
def test_collect_ranking_results(mock_fit, example_spec_modifiers):

    model, data = model_utils.model_and_data(example_spec_modifiers)
    npars = len(model.config.par_names)
    bestfit = np.asarray([0.1] * npars)
    uncertainty = np.asarray([0.01] * npars)
    # Create a 7x7 matrix filled with 0.1
    corr_mat = np.full((npars, npars), 0.1)
    # Set the diagonal to 1.0
    np.fill_diagonal(corr_mat, 1.0)
    labels = model.config.par_names
    nuisance = [label for label in labels if label != "mu"]
    fit_results = fit.FitResults(bestfit, uncertainty, labels, corr_mat, 0.0)

    results = []
    for parameter in nuisance:
        result = fit.ranking(
            model,
            data,
            fit_results=fit_results,
            parameters_list=[parameter],
            impacts_method="covariance",
        )
        results.append(result)

    combined_results = utils.collect_ranking_results(results)

    assert np.allclose(
        combined_results.bestfit, np.concatenate([result.bestfit for result in results])
    )
    assert np.allclose(
        combined_results.uncertainty,
        np.concatenate([result.uncertainty for result in results]),
    )
    assert combined_results.labels == nuisance
    assert np.allclose(
        combined_results.prefit_up,
        np.concatenate([result.prefit_up for result in results]),
    )
    assert np.allclose(
        combined_results.prefit_down,
        np.concatenate([result.prefit_down for result in results]),
    )
    assert np.allclose(
        combined_results.postfit_up,
        np.concatenate([result.postfit_up for result in results]),
    )
    assert np.allclose(
        combined_results.postfit_down,
        np.concatenate([result.postfit_down for result in results]),
    )
    assert combined_results.impacts_method == "covariance"

    # Test that the function raises a ValueError if
    # the impacts_method is not the same for all results
    results = []
    for idx, parameter in enumerate(nuisance):
        impacts_method = "auxdata_shift" if idx == 0 else "covariance"
        result = fit.ranking(
            model,
            data,
            fit_results=fit_results,
            parameters_list=[parameter],
            impacts_method=impacts_method,
        )
        assert mock_fit.call_count == 2
        results.append(result)

    with pytest.raises(
        ValueError, match="All results must be computed with same impacts method"
    ):
        utils.collect_ranking_results(results)
