"""
Provide utility functions for fitting models.
"""

from typing import List

import numpy as np

from cabinetry.fit.results_containers import RankingResults


def collect_ranking_results(results: List[RankingResults]) -> RankingResults:
    """
    collect ranking results from individual calls
    to the ranking function with subsets of the
    parameters.

    Args:
        results (List[RankingResult]): list of ranking results
        from individual calls to the ranking function.
    """
    if not all(
        result.impacts_method == results[0].impacts_method for result in results
    ):
        raise ValueError("All results must be computed with same impacts method")

    bestfit: List[float] = []
    uncertainty: List[float] = []
    labels: List[str] = []

    prefit_down: List[float] = []
    prefit_up: List[float] = []

    postfit_down: List[float] = []
    postfit_up: List[float] = []
    for result in results:
        bestfit.extend(result.bestfit)
        uncertainty.extend(result.uncertainty)
        labels.extend(result.labels)
        prefit_down.extend(result.prefit_down)
        prefit_up.extend(result.prefit_up)
        postfit_down.extend(result.postfit_down)
        postfit_up.extend(result.postfit_up)

    combined_results = RankingResults(
        np.asarray(bestfit),
        np.asarray(uncertainty),
        labels,
        np.asarray(prefit_up),
        np.asarray(prefit_down),
        np.asarray(postfit_up),
        np.asarray(postfit_down),
        results[0].impacts_method,
    )
    return combined_results
