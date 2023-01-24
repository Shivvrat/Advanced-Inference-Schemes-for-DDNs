#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import wandb
from loguru import logger

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize


@logger.catch
def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    if not cfg.DEBUG:
        wandb.init(config=cfg, project="charades_joint")
    else:
        print("We are in debug mode")
        wandb.init(config=cfg, project="charades_joint_debug")
    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)
        wandb.alert(title="Training Complete",
                    text="Completed training the model")
        import torch
        torch.cuda.empty_cache()
    # Todo: Check if same model is being used or not
    print(cfg.JOINT_LEARNING.MODEL_DIRECTORY)
    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:

        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    if cfg.DEMO.ENABLE:
        demo(cfg)


if __name__ == "__main__":
    main()
