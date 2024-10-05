"""High-level entry point to create, collect and post-process template histograms."""

import logging
import pathlib
from typing import Any, Dict, List, Optional

from cabinetry import route
from cabinetry.templates import builder
from cabinetry.templates import collector
from cabinetry.templates import postprocessor


log = logging.getLogger(__name__)


def build(
    config: Dict[str, Any],
    *,
    method: str = "uproot",
    router: Optional[route.Router] = None,
    template_list: Optional[List[route.TemplateHistogramInformation]] = None,
) -> None:
    """Produces all required histograms specified by the configuration file.

    Inputs to the histogram production are ntuples containing columnar data. Uses either
    a default method specified via ``method``, or a custom user-defined override through
    ``router``.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        method (str, optional): backend to use for histogram production, defaults to
            "uproot"
        router (Optional[route.Router], optional): instance of cabinetry.route.Router
            that contains user-defined overrides, defaults to None
        template_list (Optional[List[route.TemplateHistogramInformation]]): list of
            information for templates to process, defaults to None (all templates)
    """
    # create an instance of the class doing the template building
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    general_path = config["General"]["InputPath"]
    template_builder = builder._Builder(histogram_folder, general_path, method)

    match_func: Optional[route.MatchFunc] = None
    if router is not None:
        # specify the wrapper for user-defined functions
        router.template_builder_wrapper = template_builder._wrap_custom_template_builder
        # get a function that can be queried to return a user-defined template builder
        match_func = router._find_template_builder_match

    # get list of required templates to process if not provided already
    if template_list is None:
        template_list = route.required_templates(config)

    route.apply_to_templates(
        template_builder._create_histogram, template_list, match_func=match_func
    )


def collect(
    config: Dict[str, Any],
    *,
    method: str = "uproot",
    template_list: Optional[List[route.TemplateHistogramInformation]] = None,
) -> None:
    """Collects all required histograms specified by the configuration file.

    Histograms must already exist, and this collects and saves them in the format used
    for further processing. If no default for VariationPath is specified in the general
    settings, it defaults to an empty string.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        method (str, optional): backend to use for histogram production, defaults to
            "uproot"
        template_list (Optional[List[route.TemplateHistogramInformation]]): list of
            information for templates to process, defaults to None (all templates)
    """
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    general_path = config["General"]["InputPath"]
    variation_path = config["General"].get("VariationPath", None)
    if variation_path is None:
        # default to empty string and emit warning
        variation_path = ""
        log.info('no VariationPath specified in general settings, defaulting to ""')
    processor = collector._collector(
        histogram_folder, general_path, variation_path, method
    )

    # get list of required templates to process if not provided already
    if template_list is None:
        template_list = route.required_templates(config)

    route.apply_to_templates(processor, template_list)


def postprocess(
    config: Dict[str, Any],
    template_list: Optional[List[route.TemplateHistogramInformation]] = None,
) -> None:
    """Applies postprocessing to all histograms.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        template_list (Optional[List[route.TemplateHistogramInformation]]): list of
            information for templates to process, defaults to None (all templates)
    """
    histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])
    processor = postprocessor._postprocessor(histogram_folder)

    # get list of required templates to process if not provided already
    if template_list is None:
        template_list = route.required_templates(config)

    route.apply_to_templates(processor, template_list)
