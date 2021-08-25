from typing import List, NamedTuple

import numpy as np


class FitResults(NamedTuple):
    """Collects fit results in one object.

    Args:
        bestfit (numpy.ndarray): best-fit results of parameters
        uncertainty (numpy.ndarray): uncertainties of best-fit parameter results
        labels (List[str]): parameter labels
        corr_mat (np.ndarray): parameter correlation matrix
        best_twice_nll (float): -2 log(likelihood) at best-fit point
        goodess_of_fit (float, optional): goodness-of-fit p-value, defaults to -1
    """

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List[str]
    corr_mat: np.ndarray
    best_twice_nll: float
    goodness_of_fit: float = -1


class RankingResults(NamedTuple):
    """Collects nuisance parameter ranking results in one object.

    The best-fit results per parameter, the uncertainties, and the labels should not
    include the parameter of interest, since no impact for it is calculated.

    Args:
        bestfit (numpy.ndarray): best-fit results of parameters
        uncertainty (numpy.ndarray): uncertainties of best-fit parameter results
        labels (List[str]): parameter labels
        prefit_up (numpy.ndarray): pre-fit impact in "up" direction
        prefit_down (numpy.ndarray): pre-fit impact in "down" direction
        postfit_up (numpy.ndarray): post-fit impact in "up" direction
        postfit_down (numpy.ndarray): post-fit impact in "down" direction
    """

    bestfit: np.ndarray
    uncertainty: np.ndarray
    labels: List[str]
    prefit_up: np.ndarray
    prefit_down: np.ndarray
    postfit_up: np.ndarray
    postfit_down: np.ndarray


class ScanResults(NamedTuple):
    """Collects likelihood scan results in one object.

    Args:
        name (str): name of parameter in scan
        bestfit (float): best-fit parameter value from unconstrained fit
        uncertainty (float): uncertainty of parameter in unconstrained fit
        parameter_values (np.ndarray): parameter values used in scan
        delta_nlls (np.ndarray): -2 log(L) difference at each scanned point
    """

    name: str
    bestfit: float
    uncertainty: float
    parameter_values: np.ndarray
    delta_nlls: np.ndarray


class LimitResults(NamedTuple):
    """Collects parameter upper limit results in one object.

    Args:
        observed_limit (np.ndarray): observed limit
        expected_limit (np.ndarray): expected limit, including 1 and 2 sigma bands
        observed_CLs (np.ndarray): observed CLs values
        expected_CLs (np.ndarray): expected CLs values, including 1 and 2 sigma bands
        poi_values (np.ndarray): POI values used in scan
    """

    observed_limit: float
    expected_limit: np.ndarray
    observed_CLs: np.ndarray
    expected_CLs: np.ndarray
    poi_values: np.ndarray


class SignificanceResults(NamedTuple):
    """Collects results from a discovery significance calculation in one object.

    Args:
        observed_p_value (float): observed p-value
        observed_significance (float): observed significance
        expected_p_value (float): expected/median p-value
        expected_significance (float): expected/median significance
    """

    observed_p_value: float
    observed_significance: float
    expected_p_value: float
    expected_significance: float
