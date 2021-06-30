import logging

from . import configuration  # NOQA
from . import fit  # NOQA
from . import route  # NOQA
from . import smooth  # NOQA
from . import tabulate  # NOQA
from . import template_builder  # NOQA
from . import template_postprocessor  # NOQA
from . import model_utils  # NOQA
from . import visualize  # NOQA
from . import workspace  # NOQA


__version__ = "0.2.2"


def set_logging() -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s")
    logging.getLogger("cabinetry").setLevel(logging.DEBUG)
    logging.getLogger("pyhf").setLevel(logging.INFO)
