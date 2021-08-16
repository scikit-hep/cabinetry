"""Utility to import the Literal type."""

import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal  # noqa: F401, # pragma: no cover
else:
    from typing import Literal  # noqa: F401, # pragma: no cover
