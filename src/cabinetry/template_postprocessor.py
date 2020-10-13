import copy
import logging
import pathlib
from typing import Any, Dict

import numpy as np

from . import histo
from . import route


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram: histo.Histogram, name: str) -> None:
    """Replaces nan stat. unc. by zero for a histogram.

    Modifies the histogram handed over in the argument.

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to fix
        name (str): histogram name for logging
    """
    nan_pos = np.where(np.isnan(histogram.stdev))[0]
    if len(nan_pos) > 0:
        log.debug(f"fixing ill-defined stat. unc. for {name}")
        histogram.stdev = np.nan_to_num(histogram.stdev, nan=0.0)


def apply_postprocessing(histogram: histo.Histogram, name: str) -> histo.Histogram:
    """Returns a new histogram with post-processing applied.

    The histogram handed to the function stays unchanged. A copy of the histogram
    receives post-processing (currently only the stat. uncertainty fix) and is then
    returned.

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to postprocess
        name (str): histogram name for logging

    Returns:
        cabinetry.histo.Histogram: the fixed histogram
    """
    # copy histogram to new object to leave it unchanged
    adjusted_histogram = copy.deepcopy(histogram)
    _fix_stat_unc(adjusted_histogram, name)
    return adjusted_histogram


def _get_postprocessor(histogram_folder: pathlib.Path) -> route.ProcessorFunc:
    """Returns the post-processing function to be applied to template histograms.

    Needed by ``cabinetry.route.apply_to_all_templates``. Could alternatively create a
    ``Postprocessor`` class that contains processors.

    Args:
        histogram_folder (Union[str, pathlib.Path]): folder containing histograms

    Returns:
        route.ProcessorFunc: function to apply to a template histogram
    """

    def process_template(
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        template: str,
    ) -> None:
        """Applies post-processing to a single histogram.

        Args:
            region (Dict[str, Any]): specifying region information
            sample (Dict[str, Any]): specifying sample information
            systematic (Dict[str, Any]): specifying systematic information
            template (str): name of the template: "Nominal", "Up", "Down"
        """
        histogram = histo.Histogram.from_config(
            histogram_folder,
            region,
            sample,
            systematic,
            modified=False,
            template=template,
        )
        histogram_name = histo.build_name(region, sample, systematic, template)
        new_histogram = apply_postprocessing(histogram, histogram_name)
        histogram.validate(histogram_name)
        new_histo_path = histogram_folder / (histogram_name + "_modified")
        new_histogram.save(new_histo_path)

    return process_template


def run(config: Dict[str, Any]) -> None:
    """Applies postprocessing to all histograms.

    Args:
        config (Dict[str, Any]): cabinetry configuration
    """
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    postprocessor = _get_postprocessor(histogram_folder)
    route.apply_to_all_templates(config, postprocessor)
