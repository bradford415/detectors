import copy
import os
from pathlib import Path
from typing import Any

import yaml

INCLUDE_KEY = "__include__"


def load_config(config_path, cfg=dict()):
    """Load a primary config file and secondary additional yaml files specified inside the primary file.

    All yaml files are merged into a single configuration dictionary. This is a recursive function in case
    secondary yaml files also contain references to other yaml files.

    Args:
        config_path: path to the primary configuration yaml file which can contain references to other yaml files
    """
    assert Path(config_path).suffix in [
        ".yml",
        ".yaml",
    ], "config file must be a yaml file"

    # find the absolute path to the project root directory
    project_root = Path(__file__).resolve().parent.parent.parent

    with open(config_path) as f:
        file_cfg = yaml.safe_load(f)

    # load and merge additional yaml configs if specified in the INCLUDE_KEY field
    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY].values())
        for base_yaml in base_yamls:

            base_yaml = str(project_root / Path(base_yaml))

            with open(base_yaml) as f:
                base_cfg = load_config(base_yaml, cfg)
                merge_dict(cfg, base_cfg)

    return merge_dict(cfg, file_cfg)


def merge_dict(dict_one: dict, dict_two: dict, inplace=True) -> dict:
    """Recursively merge dictionary two into dictionary

    Dictionaries typically represent configuration parameters

    Args:
        dict_one: the dictionary to merge into
        dict_two: the dictionary to merge from
        inplace: if true, dict_one is modified in place; otherwise a deep copy of dict_one is created and modified
                 and the modified copy is returned

    Returns: the merged dictionary
    """

    def _merge(first_dict, second_dict) -> dict:
        for param in second_dict:
            # if the 2nd dict param is also a dict, and exists in dict one, recursively merge
            if (
                param in first_dict
                and isinstance(first_dict[param], dict)
                and isinstance(second_dict[param], dict)
            ):
                _merge(first_dict[param], second_dict[param])
            else:
                # once we no longer have a dict, or the dict does not exist in the first dict, we can just set the value
                first_dict[param] = second_dict[param]

        return first_dict

    if not inplace:
        dict_one = copy.deepcopy(dict_one)

    return _merge(dict_one, dict_two)


def cli_to_dict(nargs: list[str]) -> dict:
    """Parses the command line arguments and converts them to a dictionary

    Example:
        convert `a.c=3 b=10` to `{'a': {'c': 3},
                                 'b': 10}`
    Args:
        nargs: list of command line arguments
    """
    cfg = {}
    if nargs is None or len(nargs) == 0:
        return cfg

    for s in nargs:
        s = s.strip()
        k, v = s.split("=", 1)
        d = dictify(k, yaml.load(v, Loader=yaml.Loader))
        cfg = merge_dict(cfg, d)

    return cfg


def dictify(s: str, v: Any) -> dict:
    """Recursively convert a dot notation string to a nested dictionary"""
    if "." not in s:
        return {s: v}
    key, rest = s.split(".", 1)
    return {key: dictify(rest, v)}
