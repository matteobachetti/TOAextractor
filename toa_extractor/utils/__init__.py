# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions.
import base64
import io
import os

from PIL import Image


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
    """Return the root file name.

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

    if not suffix.startswith("."):
        suffix = "_" + suffix.lstrip("_")
    return newf + suffix


def encode_image_file(image_file):
    foo = Image.open(image_file)
    # Get image file
    # image_file = open(image_file, "rb")
    width, height = foo.size
    h_w_ratio = height / width
    foo.thumbnail((576, int(576 * h_w_ratio)), Image.LANCZOS)

    # From https://stackoverflow.com/questions/42503995/
    # how-to-get-a-pil-image-as-a-base64-encoded-string
    in_mem_file = io.BytesIO()
    foo.save(in_mem_file, format="JPEG")

    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()

    base64_encoded_result_bytes = base64.b64encode(img_bytes)
    base64_encoded_result_str = base64_encoded_result_bytes.decode("ascii")

    return base64_encoded_result_str


def search_substring_in_list(substring, list_of_strings):
    """
    Search for a substring in a list of strings.

    Parameters
    ----------
    substring : str
        The substring to search for.
    list_of_strings : list of str
        The list of strings to search in.

    Returns
    -------
    list
        A list of strings that contain the substring.

    Examples
    --------
    >>> search_substring_in_list('a', ['a', 'b', 'c'])
    ['a']
    >>> search_substring_in_list('a', ['a', 'b', 'c', 'ab'])
    ['a', 'ab']
    >>> search_substring_in_list('a', ['a', 'b', 'c', 'ab', 'ba'])
    ['a', 'ab', 'ba']
    """
    return [string for string in list_of_strings if substring in string]
