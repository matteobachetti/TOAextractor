# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
import pint.logging

pint.logging.setup(level="INFO")
# ----------------------------------------------------------------------------

__all__ = []
