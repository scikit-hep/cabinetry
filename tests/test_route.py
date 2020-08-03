import logging

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
            hist[...] = np.stack([yields, stdev ** 2], axis=-1)
            return hist

        return example_template_builder


@pytest.fixture
def example_callables():
    return ProcessorExamples


def test_Router():
    router = route.Router()
    assert router.template_builders == []
    assert router.template_builder_wrapper is None


def test_Router__register_processor(example_callables):
    example_router = route.Router()
    example_template_builder = example_callables.get_example_template_builder()

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

    # test registration with no details specified
    example_router._register_processor(
        example_router.template_builders,
        region_name=None,
        sample_name=None,
        systematic_name=None,
        template=None,
    )(example_template_builder)

    assert example_router.template_builders[-1] == {
        "region": "*",
        "sample": "*",
        "systematic": "*",
        "template": "*",
        "name": "example_template_builder",
        "func": example_template_builder,
    }


def test_Router_register_template_builder(example_callables):
    example_router = route.Router()
    example_template_builder = example_callables.get_example_template_builder()

    example_router.register_template_builder(
        region_name="reg", sample_name="signal", systematic_name="sys", template="Up",
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


def test_Router__find_match(example_callables, caplog):
    caplog.set_level(logging.DEBUG)
    example_router = route.Router()
    example_template_builder = example_callables.get_example_template_builder()

    def other_processor():
        pass

    example_processor_specification = {
        "region": "r?g",
        "sample": "sig*",
        "systematic": "*",
        "template": "Up",
        "name": "example_template_builder",
        "func": example_template_builder,
    }
    other_processor_specification = {
        "region": "abc",
        "sample": "*",
        "systematic": "*",
        "template": "*",
        "name": "other_processor",
        "func": other_processor,
    }
    example_router.template_builders = [
        other_processor_specification,
        example_processor_specification,
    ]

    # get a single match
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "signal", "sys", "Up"
        )
        is example_template_builder
    )

    # no matches available
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "background", "sys", "Up"
        )
        is None
    )

    # multiple matches
    caplog.clear()
    example_router.template_builders = [
        example_processor_specification,
        example_processor_specification,
    ]
    assert (
        example_router._find_match(
            example_router.template_builders, "reg", "signal", "sys", "Up"
        )
        is example_template_builder
    )

    assert (
        "found 2 matches, continuing with the first one (example_template_builder)"
        in [rec.message for rec in caplog.records]
    )
    caplog.clear()
