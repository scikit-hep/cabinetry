import fnmatch
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import boost_histogram as bh

from . import configuration


log = logging.getLogger(__name__)


# type of a function processing templates, takes sample-region-systematic-template,
# returns None
ProcessorFunc = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], str], None]

# type of a user-defined function for template processing, takes sample-region-
# systematic-template, returns a boost_histogram.Histogram
UserTemplateFunc = Callable[
    [Dict[str, Any], Dict[str, Any], Dict[str, Any], str], bh.Histogram
]

# type of a generic function that takes sample-region-systematic-template, and
# may return None or a boost_histogram.Histogram
GenericProcessorFunc = Union[ProcessorFunc, UserTemplateFunc]

# type of a function called with names of region-sample-systematic-template,
# which returns either a ProcessorFunc or None
MatchFunc = Callable[[str, str, str, str], Optional[ProcessorFunc]]

# type of wrapper function that that turns a user-defined template processing function
# (which returns a histogram) into a function that returns None
WrapperFunc = Callable[[UserTemplateFunc], ProcessorFunc]


class Router:
    """Holds user-defined processing functions and matches functions to templates.

    Provides functions for matching a pattern to apply the right function to each
    template.

    Attributes:
        template_builders (List[Dict[str, Any]]): user-defined processors for template
            building
        template_builder_wrapper (Optional[WrapperFunc]): wrapper to apply on user-
            defined template builders
    """

    def __init__(self) -> None:
        """Initialize a Router instance, with no processors or wrappers defined."""
        # initialize all lists of processor types the user can specify
        self.template_builders: List[Dict[str, Any]] = []

        # initialize the wrapper to be used for template building to turn the user-
        # defined function (which returns a histogram) into one that saves the histogram
        # this wrapper needs to be set before using _find_template_builder_match to
        # properly handle the user-defined template builder
        self.template_builder_wrapper: Optional[WrapperFunc] = None

    @staticmethod
    def _register_processor(
        processor_list: List[Dict[str, Any]],
        region_name: Optional[str],
        sample_name: Optional[str],
        systematic_name: Optional[str],
        template: Optional[str],
    ) -> Callable[[GenericProcessorFunc], GenericProcessorFunc]:
        """Decorator for registering a custom processor function.

        Args:
            region_name  (Optional[str]): name of the region to apply the function to,
                or None to apply to all regions
            sample_name  (Optional[str]): name of the sample to apply the function to,
                or None to apply to all samples
            systematic_name  (Optional[str]): name of the systematic to apply the
                function to, or None to apply to all systematics
            template (Optional[str]): name of the template to apply the function to, or
                None to apply to all templates

        Returns:
            Callable[[GenericProcessorFunc], GenericProcessorFunc]: the function to
            register a processor
        """
        if region_name is None:
            region_name = "*"
        if sample_name is None:
            sample_name = "*"
        if systematic_name is None:
            systematic_name = "*"
        if template is None:
            template = "*"

        def _register(func: GenericProcessorFunc) -> GenericProcessorFunc:
            """Registers a processor function to be applied when matching a pattern.

            The pattern is specified by region-sample-systematic-template.

            Args:
                func (GenericProcessorFunc): the function to register
            """
            processor_list.append(
                {
                    "region": region_name,
                    "sample": sample_name,
                    "systematic": systematic_name,
                    "template": template,
                    "name": func.__name__,
                    "func": func,
                }
            )
            return func

        return _register

    def register_template_builder(
        self,
        region_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        systematic_name: Optional[str] = None,
        template: Optional[str] = None,
    ) -> Callable[[Callable], UserTemplateFunc]:
        """Decorator for registering a template builder function.

        Args:
            region_name (Optional[str], optional): name of the region to apply the
                function to, defaults to None (apply to all regions)
            sample_name (Optional[str], optional): name of the sample to apply the
                function to, defaults to None (apply to all samples)
            systematic_name (Optional[str], optional): name of the systematic to apply
                the function to, defaults to None (apply to all systematics)
            template (Optional[str], optional): name of the template to apply the
                function to, defaults to None (apply to all templates)

        Returns:
            Callable[[Callable], UserTemplateFunc]: the generic function to register a
            processor
        """
        return self._register_processor(
            self.template_builders, region_name, sample_name, systematic_name, template
        )

    @staticmethod
    def _find_match(
        processor_list: List[Dict[str, Any]],
        region_name: str,
        sample_name: str,
        systematic_name: str,
        template: str,
    ) -> Optional[GenericProcessorFunc]:
        """Returns a function matching the provided specification.

        Args:
            processor_list (List[Dict[str, Any]]): list of processors to search in
            region_name (str): region name
            sample_name (str): sample name
            systematic_name (str): systematic name
            template (str): template name

        Returns:
            Optional[GenericProcessorFunc]: processor function matching the description,
            or None if no matches are found
        """
        matches = []
        for processor in processor_list:
            region_matches = fnmatch.fnmatch(region_name, processor["region"])
            sample_matches = fnmatch.fnmatch(sample_name, processor["sample"])
            systematic_matches = fnmatch.fnmatch(
                systematic_name, processor["systematic"]
            )
            template_matches = fnmatch.fnmatch(template, processor["template"])
            if (
                region_matches
                and sample_matches
                and systematic_matches
                and template_matches
            ):
                matches.append(processor["func"])

        if len(matches) > 1:
            log.warning(
                f"found {len(matches)} matches, continuing with the first one "
                f"({matches[0].__name__})"
            )
        elif len(matches) == 0:
            return None
        return matches[0]

    def _find_template_builder_match(
        self, region_name: str, sample_name: str, systematic_name: str, template: str
    ) -> Optional[ProcessorFunc]:
        """Returns a template builder function matching the provided specification.

        If no matches are found, returns None. Wraps the user-defined function in the
        provided wrapper, if no wrapper is found raises an error.

        Args:
            region_name (str): region name
            sample_name (str): sample name
            systematic_name (str): systematic name
            template (str): template name

        Raises:
            ValueError: when no template wrapper is set

        Returns:
            Optional[ProcessorFunc]: wrapped template builder function matching the
            description, or None if no matches are found
        """
        if self.template_builder_wrapper is None:
            # a wrapper needs to be defined already to convert the user-defined
            # UserTemplateFunc into a ProcessorFunc
            raise ValueError("no template builder wrapper defined")

        match = self._find_match(
            self.template_builders, region_name, sample_name, systematic_name, template
        )

        if match is not None:
            # if user-defined function was found, wrap and return it
            return self.template_builder_wrapper(match)
        return None


def apply_to_all_templates(
    config: Dict[str, Any],
    default_func: ProcessorFunc,
    match_func: Optional[MatchFunc] = None,
) -> None:
    """Applies the supplied function ``default_func`` to all templates.

    The templates are specified by the configuration file. The function takes four
    arguments in this order:

    - the dict specifying region information
    - the dict specifying sample information
    - the dict specifying systematic information
    - name of the template being considered: "Nominal", "Up", "Down"

    In addition it is possible to specify a function that returns custom overrides. If
    one is found for a given template, it is used instead of the default.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        default_func (ProcessorFunc): function to be called for every template by
            default
        match_func: (Optional[MatchFunc], optional): function that returns user-defined
            functions to override the call to ``default_func``, defaults to None (then
            it is not used)
    """
    for region in config["Regions"]:
        log.debug(f"  in region {region['Name']}")

        for sample in config["Samples"]:
            log.debug(f"    reading sample {sample['Name']}")

            for systematic in [{"Name": "Nominal"}] + config["Systematics"]:

                # determine how many templates need to be considered
                if systematic["Name"] == "Nominal":
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
                        # no further action is needed, continue with the next
                        # region-sample-systematic combination
                        continue

                    log.debug(
                        f"      variation {systematic['Name']}"
                        f"{' ' + template if template != 'Nominal' else ''}"
                    )

                    func_override = None
                    if match_func is not None:
                        # check whether a user-defined function was registered that
                        # matches this region-sample-systematic-template
                        func_override = match_func(
                            region["Name"], sample["Name"], systematic["Name"], template
                        )
                    if func_override is not None:
                        # call the user-defined function
                        log.debug(
                            f"executing user-defined override "
                            f"{func_override.__name__}"
                        )
                        func_override(region, sample, systematic, template)
                    else:
                        # call the provided default function
                        default_func(region, sample, systematic, template)
