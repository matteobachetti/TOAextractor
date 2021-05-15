import os

from astropy import log
import yaml
from yaml import load, Loader
from functools import lru_cache

NOT_AVLB = "N/A"


def parse_config_dict(config):
    """
    Examples
    --------
    >>> config = {'a': 1, 'b': 'B', 'c': {'ca': 2, 'cb': "NOT_AVLB"}}
    >>> assert parse_config_dict(config) == {'a': 1, 'b': 'B', 'c': {'ca': 2, 'cb': NOT_AVLB}}
    """
    for key, value in config.items():
        if value == "NOT_AVLB":
            config[key] = NOT_AVLB
        elif isinstance(value, dict):
            config[key] = parse_config_dict(value)
    return config


@lru_cache(maxsize=32)
def read_config(config_file):
    """Read configuration file."""
    if config_file == "default" and not os.path.exists(config_file):
        return {"timeout": 10}
    with open(config_file) as fobj:
        try:
            config = load(fobj, Loader=Loader)
        except yaml.scanner.ScannerError as e:
            log.error(str(e))
    return parse_config_dict(config)


def _get_mission_template(template_info, info):
    if info["mission"] not in template_info:
        log.info(f"No specific template for {info['mission']}. Using default")
        return template_info["default"]
    template_info = template_info[info["mission"]]
    if isinstance(template_info, str):
        return template_info
    else:
        return template_info["default"]


def get_template(source, info=None):
    source = source.lower()
    curdir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(curdir, "..", "data")
    template_file = os.path.join(template_dir, "templates.yaml")
    template_info = load_yaml_file(template_file)

    for src in template_info:
        if src.lower() in source:
            template = _get_mission_template(template_info[src], info)
            return os.path.join(template_dir, template)
    else:
        log.error(f"Template for {source} not found; using Crab")
        return os.path.join(template_dir, template_info["crab"]["default"])


def load_yaml_file(infofile):
    with open(infofile) as fobj:
        return yaml.load(fobj, Loader=yaml.Loader)
