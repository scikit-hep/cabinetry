import copy
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import configuration
from . import histo


log = logging.getLogger(__name__)


def _fix_stat_unc(histogram: histo.H, name: str) -> None:
    """replace nan stat. unc. by zero for a histogram, modifies the
    histogram handed over in the argument

    Args:
        histogram (cabinetry.histo.Histogram): the histogram to fix
        name (str): histogram name for logging
    """
    nan_pos = np.where(np.isnan(histogram.stdev))[0]
    if len(nan_pos) > 0:
        log.debug(f"fixing ill-defined stat. unc. for {name}")
        histogram.stdev = np.nan_to_num(histogram.stdev, nan=0.0)


def apply_postprocessing(histogram: histo.H, name: str) -> histo.H:
    """Create a new modified histogram, currently only calling the
    stat. uncertainty fix. The histogram in the function argument
    stays unchanged.

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


def run(config: Dict[str, Any], histogram_folder: str) -> None:
    """apply post-processing as needed for all histograms
    this is very similar to template_builder.create_histograms() and should be refactored

    Args:
        config (Dict[str, Any]): cabinetry configuration
        histogram_folder (str): folder containing the histograms
    """
    log.info("applying post-processing to histograms")
    # loop over all histograms
    for region in config["Regions"]:
        for sample in config["Samples"]:
            for systematic in [{"Name": "nominal"}] + config["Systematics"]:
                # determine how many templates need to be considered
                if systematic["Name"] == "nominal":
                    # only nominal template is needed
                    templates = ["Nominal"]
                else:
                    # systematics can have up and down template
                    templates = ["Up", "Down"]
                for template in templates:
                    # determine whether a histogram is needed for this
                    # specific combination of sample-region-systematic-template
                    histo_needed = configuration.histogram_is_needed(
                        region, sample, systematic, template
                    )

                    if not histo_needed:
                        # no further action is needed, continue with the next region-sample-systematic combination
                        continue

                    histogram = histo.Histogram.from_config(
                        histogram_folder,
                        region,
                        sample,
                        systematic,
                        modified=False,
                        template=template,
                    )
                    histogram_name = histo.build_name(
                        region, sample, systematic, template
                    )
                    new_histogram = apply_postprocessing(histogram, histogram_name)
                    histogram.validate(histogram_name)
                    new_histo_path = Path(histogram_folder) / (
                        histogram_name + "_modified"
                    )
                    new_histogram.save(new_histo_path)
