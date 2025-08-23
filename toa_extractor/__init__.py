# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
import warnings
from pint.utils import check_longdouble_precision

from ._astropy_init import *  # noqa

# ----------------------------------------------------------------------------

__all__ = []


if not check_longdouble_precision():
    warnings.warn(
        "TOAextractor requires longdouble precision to be at least 80 bits. "
        "This is not the case on your system. It is often possible to fix this by "
        "installing a different Python interpreter. For ARM-based Macs, follow the instructions at "
        "https://nanograv-pint.readthedocs.io/en/latest/installation.html#apple-m1-m2-processors",
        RuntimeWarning,
    )
