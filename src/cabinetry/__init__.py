"""The cabinetry library."""

import logging
from typing import List

import cabinetry.configuration as configuration  # noqa: F401
import cabinetry.fit as fit  # noqa: F401
import cabinetry.histo as histo  # noqa: F401
import cabinetry.model_utils as model_utils  # noqa: F401
import cabinetry.route as route  # noqa: F401
import cabinetry.smooth as smooth  # noqa: F401
import cabinetry.tabulate as tabulate  # noqa: F401
import cabinetry.templates as templates  # noqa: F401
import cabinetry.visualize as visualize  # noqa: F401
import cabinetry.workspace as workspace  # noqa: F401


__all__ = [
    "__version__",
    "set_logging",
    "configuration",
    "fit",
    "histo",
    "model_utils",
    "route",
    "smooth",
    "tabulate",
    "templates",
    "visualize",
    "workspace",
]


def __dir__() -> List[str]:
    return __all__


__version__ = "0.5.2"


def set_logging() -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s")
    logging.getLogger("cabinetry").setLevel(logging.DEBUG)
    logging.getLogger("pyhf").setLevel(logging.INFO)
