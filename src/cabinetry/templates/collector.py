"""Collects required template histograms provided by user."""

import logging
import pathlib
from typing import Any, cast, Dict, Optional

from cabinetry import histo
from cabinetry import route
from cabinetry._typing import Literal
from cabinetry.templates import utils


log = logging.getLogger(__name__)


def _histo_path(
    general_path: str,
    variation_path: str,
    region: Dict[str, Any],
    sample: Dict[str, Any],
    systematic: Dict[str, Any],
    template: Optional[Literal["Up", "Down"]],
) -> str:
    """Returns the paths to a histogram for a region-sample-systematic-template.

    The path should contain a colon to separate the path to the file itself and the
    location of the histogram within the file. Due to the presence of this colon, the
    return value is a string instead of a pathlib path. A path is built starting from
    the path specified in the general options in the configuration file. This path
    contains placeholders for region-, sample-, and systematic-specific values.

    Args:
        general_path (str): path specified in general settings, with sections that can
            be overridden by region / sample / systematic settings
        variation_path (str): default for VariationPath setting
        region (Dict[str, Any]): containing all region information
        sample (Dict[str, Any]): containing all sample information
        systematic (Dict[str, Any]): containing all systematic information
        template (Optional[Literal["Up", "Down"]]): template considered: "Up", "Down",
            or None for nominal

    Raises:
        ValueError: when ``RegionPath`` placeholder is used, but region setting is not
            specified
        ValueError: when ``SamplePath`` placeholder is used, but sample setting is not
            specified

    Returns:
        str: path to input file (and histogram within)
    """
    # obtain region and sample paths, if they are defined
    region_path = region.get("RegionPath", None)
    sample_path = sample.get("SamplePath", None)

    # check whether a systematic is being processed
    if template is not None:
        # determine whether the template has an override for VariationPath specified
        variation_override = utils._check_for_override(
            systematic, template, "VariationPath"
        )
        if variation_override is not None:
            # _check_for_override should return Optional[str] for VariationPath, but
            # mypy cannot know that it will not be a list, so explicitly cast to str
            variation_path = cast(str, variation_override)
        else:
            log.warning(
                f"no VariationPath override specified for {region['Name']} / "
                f"{sample['Name']} / {systematic['Name']} {template}"
            )
    # apply variation-specific setting
    path = general_path.replace("{VariationPath}", variation_path)

    # handle region-specific setting
    region_template_exists = "{RegionPath}" in general_path
    if region_path is not None:
        if not region_template_exists:
            log.warning(
                "region override specified, but {RegionPath} not found in default path"
            )
        path = path.replace("{RegionPath}", region_path)
    elif region_template_exists:
        raise ValueError(f"no path setting found for region {region['Name']}")

    # handle sample-specific setting
    sample_template_exists = "{SamplePath}" in general_path
    if sample_path is not None:
        if not sample_template_exists:
            log.warning(
                "sample override specified, but {SamplePath} not found in default path"
            )
        path = path.replace("{SamplePath}", sample_path)
    elif sample_template_exists:
        raise ValueError(f"no path setting found for sample {sample['Name']}")

    # check for presence of colon to distinguish path to file and location within file
    if ":" not in path:
        log.warning(f"no colon found in path {path}, may not be able to find histogram")

    return path


def _collector(
    histogram_folder: pathlib.Path, general_path: str, variation_path: str, method: str
) -> route.ProcessorFunc:
    """Returns the histogram-collecting function to be applied to template histograms.

    Needed by ``cabinetry.route.apply_to_all_templates``. Could alternatively create a
    ``Collector`` class that contains processors (see ``builder._Builder`` for an
    example).

    Args:
        histogram_folder (pathlib.Path): folder to save the histograms to
        general_path (str): template for paths to input files for histogram building
        variation_path (str): default for VariationPath setting
        method (str): backend to use for histogram production

    Raises:
            NotImplementedError: when requesting an unknown backend

    Returns:
        route.ProcessorFunc: function to apply to a template histogram
    """

    def collect_template(
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
        template: Optional[Literal["Up", "Down"]],
    ) -> None:
        """Collects a histogram and writes it to a file.

        Args:
            region (Dict[str, Any]): containing all region information
            sample (Dict[str, Any]): containing all sample information
            systematic (Dict[str, Any]): containing all systematic information
            template (Optional[Literal["Up", "Down"]]): template considered: "Up",
                "Down", or None for nominal
        """
        histo_path = _histo_path(
            general_path, variation_path, region, sample, systematic, template
        )

        # obtain the histogram
        if method == "uproot":
            from cabinetry.contrib import histogram_reader

            histogram = histogram_reader.with_uproot(histo_path)

        else:
            raise NotImplementedError(f"unknown backend {method}")

        # store information in a Histogram instance and save it
        utils._name_and_save(
            histogram_folder,
            histo.Histogram(histogram),
            region,
            sample,
            systematic,
            template,
        )

    return collect_template
