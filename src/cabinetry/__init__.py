import logging

import cabinetry.configuration  # noqa: F401
import cabinetry.fit  # noqa: F401
import cabinetry.model_utils  # noqa: F401
import cabinetry.route  # noqa: F401
import cabinetry.smooth  # noqa: F401
import cabinetry.tabulate  # noqa: F401
import cabinetry.template_builder  # noqa: F401
import cabinetry.template_postprocessor  # noqa: F401
import cabinetry.visualize  # noqa: F401
import cabinetry.workspace  # noqa: F401


__version__ = "0.3.0"


def set_logging() -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s")
    logging.getLogger("cabinetry").setLevel(logging.DEBUG)
    logging.getLogger("pyhf").setLevel(logging.INFO)
