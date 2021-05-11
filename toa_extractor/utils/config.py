import glob
import os
import sys
import shutil
import tempfile
import re
import datetime
import subprocess as sp

from astropy import log
import yaml
from yaml import load, Loader, Dumper
from functools import lru_cache
NOT_AVLB = 'N/A'


def parse_config_dict(config):
    """
    Examples
    --------
    >>> config = {'a': 1, 'b': 'B', 'c': {'ca': 2, 'cb': "NOT_AVLB"}}
    >>> assert parse_config_dict(config) == {'a': 1, 'b': 'B', 'c': {'ca': 2, 'cb': NOT_AVLB}}
    """
    for key, value in config.items():
        if value == 'NOT_AVLB':
            config[key] = NOT_AVLB
        elif isinstance(value, dict):
            config[key] = parse_config_dict(value)
    return config


@lru_cache(maxsize=32)
def read_config(config_file):
    """Read configuration file."""
    if config_file == 'default' and not os.path.exists(config_file):
        return {'timeout': 10}
    with open(config_file) as fobj:
        try:
            config = load(fobj, Loader=Loader)
        except yaml.scanner.ScannerError as e:
            log.error(str(e))
    return parse_config_dict(config)
