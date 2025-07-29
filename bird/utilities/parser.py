import json
import logging
import os
from pathlib import Path

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


def check_for_tabs_in_yaml(file_path: str) -> None:
    """
    Checks if a YAML file contains any tab characters.
    Raises a ValueError if tabs found.

    Parameters
    ----------
    file_path: str
        path to yaml file
    """

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for iline, line in enumerate(lines):
        if "\t" in line:
            error_msg = (
                f"Tab character found on line {iline} of '{file_path}'. "
            )
            error_msg += "\nYAML files must use spaces for indentation."
            logger.error(error_msg)
            raise ValueError(error_msg)


def parse_json(file_path: str) -> dict:
    """
    Parse a json file into a dictionary

    Parameters
    ----------
    file_path: str
        path to json file

    Returns
    ----------
    inpt: dict
        Dictionary that contains the json file content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path) as f:
        inpt = json.load(f)
    return inpt


def parse_yaml(file_path: str) -> dict:
    """
    Parse a yaml file into a dictionary

    Parameters
    ----------
    file_path: str
        path to yaml file

    Returns
    ----------
    inpt: dict
        Dictionary that contains the yaml file content
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    # Make sure the yaml file does not contain tabs
    check_for_tabs_in_yaml(file_path)

    # Parse
    yaml = YAML(typ="safe")
    inpt = yaml.load(Path(file_path))
    return inpt
