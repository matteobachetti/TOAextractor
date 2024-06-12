# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.
import os


def safe_get_key(dictionary, keys, default):
    """
    Examples
    --------
    >>> d = dict(a=2, b=2, c=dict(d=12, e=45, f=dict(g=32)))
    >>> safe_get_key(d, 'a', 1)
    2
    >>> safe_get_key(d, ['a'], 1)
    2
    >>> safe_get_key(d, ['asdf'], 1)
    1
    >>> safe_get_key(d, ['c', 'f', 'g'], 1)
    32
    >>> safe_get_key(d, ['c', 'f', 'asdfasfd'], 1)
    1
    >>> safe_get_key(d, ['c', 'fasdfasf', 'g'], 1)
    1
    """
    if isinstance(keys, str):
        keys = [keys]

    current_key = keys.pop(0)
    try:
        if len(keys) == 0:
            return dictionary[current_key]
        else:
            return safe_get_key(dictionary[current_key], keys, default)
    except KeyError:
        return default


def root_name(filename):
    """Return the root file name (without _ev, _lc, etc.).

    Parameters
    ----------
    filename : str

    Examples
    --------
    >>> root_name("file.evt.gz")
    'file'
    >>> root_name("file.ds")
    'file'
    >>> root_name("file.1.ds")
    'file.1'
    >>> root_name("file.1.ds.gz")
    'file.1'
    >>> root_name("file.1.ds.gz.Z")
    'file.1'
    """
    fname = filename
    while os.path.splitext(fname)[1] in [".gz", ".Z", ".zip", ".bz"]:
        fname = fname.replace(os.path.splitext(fname)[1], "")
    fname = os.path.splitext(fname)[0]
    return fname


def output_name(filename, version, suffix):
    """Create an output file name, prepending a single underscore to each part.

    Parameters
    ----------
    filename : str

    Examples
    --------
    >>> output_name("file.evt.gz", "v3", "folded.h5")
    'file_v3_folded.h5'
    >>> output_name("file.evt.gz", "v3", "__folded.h5")
    'file_v3_folded.h5'
    """
    newf = root_name(filename)
    if version is not None or version != "none":
        newf += "_" + version.lstrip("_")
    return newf + "_" + suffix.lstrip("_")
