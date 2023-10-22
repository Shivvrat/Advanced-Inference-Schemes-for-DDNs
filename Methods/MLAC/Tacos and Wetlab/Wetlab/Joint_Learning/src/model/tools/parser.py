#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import itertools
import sys
from pathlib import Path

from tools.default_config import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description="Dependency Network")
    # Train
    # parser.add_argument('--MODEL.EPOCHS',
    #                     type=int,
    #                     help='Number of epochs used to train the model')
    # parser.add_argument('--MODEL.LEARNING_RATE',
    #                     type=float,
    #                     help='Learning rate used to train the model')
    # parser.add_argument('--MODEL.L1_WEIGHT',
    #                     type=float,
    #                     help='L1 regularization weight used to train the model')
    # parser.add_argument('--SAMPLING.NUM_SAMPLES',
    #                     type=int,
    #                     help='Number of samples used')
    # parser.add_argument('--SAMPLING.THRESHOLD',
    #                     type=float,
    #                     help='Threshold used')
    # parser.add_argument('--MODEL.DIRECTORY',
    #                     type=str,
    #                     help='Locations where models are stored')
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


def load_cfgs(args):
    # Priority 3: get default charades
    # Setup cfg.
    cfg_base = get_cfg_defaults()

    # Priority 2: merge from yaml config
    yaml_path = cfg_base.CONFIG_PATH
    if yaml_path is not None and Path(yaml_path).exists():
        cfg_base.merge_from_file(yaml_path)

    # Priority 1: merge from .env
    # cfg_base.merge_from_list(args)
    if args is not None and args.opts is not None:
        cfg_base.merge_from_list(args.opts)

    # if hasattr(args, "MODEL.DIRECTORY") :
    #     cfg_base.MODEL.DIRECTORY = args.MODEL.DIRECTORY


    return cfg_base
