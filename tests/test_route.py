import functools
import logging
from unittest import mock

import boost_histogram as bh
import numpy as np
import pytest

from cabinetry import route


class ProcessorExamples:
    @staticmethod
    def get_example_template_builder():
        def example_template_builder(
            reg: dict, sam: dict, sys: dict, tem: str
        ) -> bh.Histogram:
            hist = bh.Histogram(bh.axis.Variable([0, 1]), storage=bh.storage.Weight())
            yields = np.asarray([2])
            stdev = np.asarray([0.1])
            hist[...] = np.stack([yields, stdev**2], axis=-1)
            return hist

        return example_template_builder


@pytest.fixture
def processor_examples():
    return ProcessorExamples


def test_Router():
    router = route.Router()
    assert router.template_builders == []
    assert router.template_builder_wrapper is None


def test_Router__register_processor(processor_examples):
    example_router = route.Router()
    example_template_builder = processor_examples.get_example_template_builder()

    example_router._register_processor(
        example_router.template_builders,
        region_name="reg",
        sample_name="signal",
        systematic_name="sys",
        template="Up",
    )(example_template_builder)

    assert example_router.template_builders == [
        {
            "region": "reg",
            "sample": "signal",
            "systematic": "sys",
            "template": "Up",
            "name": "example_template_builder",
            "func": example_template_builder,
        }
    ]


def test_Router_register_template_builder(processor_examples):
    example_router = route.Router()
    example_template_builder = processor_examples.get_example_template_builder()

    example_router.register_template_builder(
        region_name="reg", sample_name="signal", systematic_name="sys", template="Up"
    )(example_template_builder)

    assert example_router.template_builders == [
        {
            "region": "reg",
            "sample": "signal",
            "systematic": "sys",
            "template": "Up",
            "name": "example_template_builder",
            "func": example_template_builder,
        }
    ]

    # defaults for arguments
    example_router.register_template_builder()(example_template_builder)
    assert example_router.template_builders[1] == {
        "region": "*",
        "sample": "*",
        "systematic": "*",
        "template": "*",
        "name": "example_template_builder",
        "func": example_template_builder,
    }


def test_Router__find_match(processor_examples, caplog):
    caplog.set_level(logging.DEBUG)
    example_router = route.Router()
    example_template_builder_1 = processor_examples.get_example_template_builder()
    example_template_builder_2 = processor_examples.get_example_template_builder()
    example_template_builder_3 = processor_examples.get_example_template_builder()

    processor_specification_1 = {
        "region": "r?g",
        "sample": "sig*",
        "systematic": "*",
        "template": "Up",
        "name": "example_template_builder_1",
        "func": example_template_builder_1,
    }
    processor_specification_2 = {
        "region": "abc",
        "sample": "*",
        "systematic": "*",
        "template": "*",
        "name": "example_template_builder_2",
        "func": example_template_builder_2,
    }
    processor_specification_3 = {
        "region": "reg",
        "sample": "bg",
        "systematic": "*",
        "template": None,
        "name": "example_template_builder_3",
        "func": example_template_builder_3,
    }
    example_router.template_builders = [
        processor_specification_1,
        processor_specification_2,
        processor_specification_3,
    ]

    # get a single match
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "signal", "sys", "Up"
        )
        is example_template_builder_1
    )

    # strings match case 3, but template does not (processor only applied to nominal)
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "bg", "sys", "Up"
        )
        is None
    )

    # no matches available due to string mismatch
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "background", "sys", "Up"
        )
        is None
    )

    # match for None (nominal) template with no processor template restrictions ("*")
    assert (
        example_router._find_match(
            example_router.template_builders, "abc", "signal", "sys", None
        )
        is example_template_builder_2
    )

    # match for None (nominal) template with None processor template
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "bg", "*", None
        )
        is example_template_builder_3
    )

    # multiple matches
    caplog.clear()
    example_router.template_builders = [
        processor_specification_1,
        processor_specification_1,
    ]
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "signal", "sys", "Up"
        )
        is example_template_builder_1
    )

    assert (
        "found 2 matches, continuing with the first one (example_template_builder)"
        in [rec.message for rec in caplog.records]
    )
    caplog.clear()


def test_Router__find_template_builder_match(processor_examples):
    example_router = route.Router()
    example_template_builder = processor_examples.get_example_template_builder()

    # no wrapper defined
    with pytest.raises(ValueError, match="no template builder wrapper defined"):
        example_router._find_template_builder_match("", "", "", None)

    def example_wrapper(func):
        @functools.wraps(func)
        def wrapper(reg, sam, sys, tem):
            # return the bin yield of the histogram to have something to compare
            return func(reg, sam, sys, tem).values()

        return wrapper

    # wrapper defined, but no match available
    example_router.template_builder_wrapper = example_wrapper
    assert example_router._find_template_builder_match("", "", "", None) is None

    # match exists
    with mock.patch(
        "cabinetry.route.Router._find_match", return_value=example_template_builder
    ) as mock_find:
        wrapped_builder = example_router._find_template_builder_match(
            "reg", "", "", None
        )
        assert mock_find.call_args_list == [(([], "reg", "", "", None), {})]

        # need to verify that wrapped template builder is wrapped with right function
        expected_wrap = example_wrapper(example_template_builder)
        assert wrapped_builder.__name__ == expected_wrap.__name__

        # a direct assert of equality fails
        # assert wrapped_builder == expected_wrap

        # compare instead the behavior for an example call
        assert wrapped_builder({}, {}, {}, {}) == expected_wrap({}, {}, {}, {})


def test_apply_to_all_templates():
    # could mock configuration.histogram_is_needed here
    # define a custom override function that logs its arguments when called
    override_call_args = []

    def match_func(reg: str, sam: str, sys: str, tem: str):
        def f(reg: dict, sam: dict, sys: dict, tem: str):
            override_call_args.append((reg, sam, sys, tem))

        return f

    default_func = mock.MagicMock()

    example_config = {
        "Regions": [{"Name": "test_region"}],
        "Samples": [{"Name": "sample"}],
        "Systematics": [
            {"Name": "norm", "Type": "Normalization"},
            {"Name": "var", "Type": "NormPlusShape"},
        ],
    }

    # no overrides specified
    route.apply_to_all_templates(example_config, default_func, match_func=None)

    # check that the default function was called for all templates
    assert default_func.call_count == 3
    assert default_func.call_args_list[0] == (
        ({"Name": "test_region"}, {"Name": "sample"}, {}, None),
        {},
    )
    assert default_func.call_args_list[1] == (
        (
            {"Name": "test_region"},
            {"Name": "sample"},
            {"Name": "var", "Type": "NormPlusShape"},
            "Up",
        ),
        {},
    )
    assert default_func.call_args_list[2] == (
        (
            {"Name": "test_region"},
            {"Name": "sample"},
            {"Name": "var", "Type": "NormPlusShape"},
            "Down",
        ),
        {},
    )

    # apply the override instead to all templates
    route.apply_to_all_templates(example_config, default_func, match_func=match_func)
    assert len(override_call_args) == 3
    assert override_call_args[0] == (
        {"Name": "test_region"},
        {"Name": "sample"},
        {},
        None,
    )
    assert override_call_args[1] == (
        {"Name": "test_region"},
        {"Name": "sample"},
        {"Name": "var", "Type": "NormPlusShape"},
        "Up",
    )
    assert override_call_args[2] == (
        {"Name": "test_region"},
        {"Name": "sample"},
        {"Name": "var", "Type": "NormPlusShape"},
        "Down",
    )

    # no systematics
    example_config = {
        "Regions": [{"Name": "test_region"}],
        "Samples": [{"Name": "sample"}],
    }
    route.apply_to_all_templates(example_config, default_func)
    # previously 3 calls of default_func, now one more for nominal template
    assert default_func.call_count == 4
    assert default_func.call_args_list[3][0][3] is None
