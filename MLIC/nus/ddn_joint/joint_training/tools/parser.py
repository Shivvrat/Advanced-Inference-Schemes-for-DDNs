#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys
from pathlib import Path

from tools.default_config import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description="Dependency Network")
    parser.add_argument(
            "opts",
            help="See default_config.py for all options, extra args to pass to the command (optional)",
            default=None,
            nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def convert_to_dict(cfg_node, key_list = None):
    """ Convert a config node to dictionary """
    if key_list is None:
        key_list = []
    from yacs.config import CfgNode, _VALID_TYPES
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                f'Key {".".join(key_list)} with value {type(cfg_node)} is not a valid type; valid types: {_VALID_TYPES}')

        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def load_cfgs(args, config_path):
    # Priority 3: get default wetlab
    # Setup cfg.
    cfg_base = get_cfg_defaults()

    # Priority 2: merge from yaml config
    # yaml_path = cfg_base.CONFIG_PATH
    yaml_path = config_path
    if yaml_path is not None and Path(yaml_path).exists():
        print(f"We loaded the yaml file {yaml_path}")
        cfg_base.merge_from_file(yaml_path)
    else:
        print("yaml file not found")

    # Priority 1: merge from .env
    # cfg_base.merge_from_list(args)
    if args is not None and args.opts is not None:
        cfg_base.merge_from_list(args.opts)

    # if hasattr(args, "MODEL.DIRECTORY") :
    #     cfg_base.MODEL.DIRECTORY = args.MODEL.DIRECTORY


    return cfg_base
