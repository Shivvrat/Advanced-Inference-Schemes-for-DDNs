import os

from default_config import get_cfg_defaults
from pathlib import Path

# from dotenv import load_dotenv, find_dotenv
import itertools


def combine_cfgs(yaml_path, args):
    # Priority 3: get default charades
    cfg_base = get_cfg_defaults()

    # Priority 2: merge from yaml config
    if yaml_path is not None and Path(yaml_path).exists():
        cfg_base.merge_from_file(yaml_path)

    # Priority 1: merge from .env
    list_of_args = list(vars(args).items())
    cfg_base.merge_from_list(list(itertools.chain(*list_of_args)))
    # load_dotenv(find_dotenv(), verbose=True) # Load .env

    # Load variables

    return cfg_base
