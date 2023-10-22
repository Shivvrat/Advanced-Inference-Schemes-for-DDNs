#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

import wandb
from dn.learn_dn import train_dn_one_iter, save_model, val_dn_one_iter
from dn.get_dn_models import get_model

logger = logging.get_logger(__name__)


def train_epoch(
        train_loader,
        model,
        dn_models,
        optimizer,
        scaler,
        train_meter,
        cur_epoch,
        cfg,
        lr,
        writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    dn_models.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    dn_total_loss = 0

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
                mixup_alpha=cfg.MIXUP.ALPHA,
                cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
                mix_prob=cfg.MIXUP.PROB,
                switch_prob=cfg.MIXUP.SWITCH_PROB,
                label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
                num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        new_lr = optim.get_epoch_lr(
            cur_epoch + float(cur_iter) / data_size, cfg)
        if new_lr != lr:
            optim.set_lr(optimizer, lr)
            if cfg.JOINT_LEARNING.PRETRAINED:
                dn_models.set_lr(lr)
            else:
                dn_models.set_lr(lr*10)
            lr = new_lr
        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, meta["boxes"])
            else:
                preds = model(inputs)
            # Explicitly declare reduction to mean.
            loss_fun_1 = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(
                    reduction="mean"
            )
            # Compute the loss.
            loss_1 = loss_fun_1(preds, labels)
        # wandb.log({'loss': loss_1})
        # Get gradients from dn_lr
        gradients_from_dn, dn_models, dn_loss = train_dn_one_iter(preds.detach().clone().float(), labels.clone().float(), dn_models,
                                                                  action="average")
        dn_total_loss += dn_loss
        # with torch.no_grad():
        #     # final_outputs = torch.Tensor(final_outputs)
        #     loss_2 = loss_fun_1(preds, labels)
        optimizer.zero_grad()

        # check Nan Loss.
        misc.check_nan_losses(loss_1)
        if not cfg.JOINT_LEARNING.TWO_LOSSES:
            # Only using loss from DN
            preds.backward(gradients_from_dn)

            # scaler.scale(loss_1).backward()
            # scaler.unscale_(optimizer)

        # Only use the loss from DN
        else:
            # # My code for joint learning
            # scaler.scale(preds).backward(gradients_from_dn)
            # scaler.unscale_(optimizer)
            # Using two losses here - one from DN and one from CNN
            preds.backward(gradients_from_dn, retain_graph=True)
            loss_1.backward()
            # scaler.scale(loss_1).backward()
            # scaler.unscale_(optimizer)

        # Unscales the gradients of optimizer's assigned params in-place
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                    labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss_1 = du.all_reduce([loss_1])[0]
            loss_1 = loss_1.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss_1, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                        {"Train/loss": loss_1, "Train/lr": lr},
                        global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss_1] = du.all_reduce([loss_1])
                loss_1 = loss_1.item()

            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(
                    preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_1, top1_err, top5_err = du.all_reduce(
                            [loss_1, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_1, top1_err, top5_err = (
                    loss_1.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss_1,
                    lr,
                    inputs[0].size(0)
                    * max(
                            cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                        {
                            "Train/loss_1": loss_1,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    this_loss = train_meter.loss_total
    num_iter = len(train_loader)
    wandb.log({'cnn train loss': this_loss/num_iter,
               'dn train loss': dn_total_loss/num_iter})
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return dn_models


@torch.no_grad()
def eval_epoch(val_loader, model, dn_models, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        dn_models:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(
                    du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(
                    preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                        top1_err,
                        top5_err,
                        inputs[0].size(0)
                        * max(
                                cfg.NUM_GPUS, 1
                        ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                            {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                            global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                    {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                    preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    if not cfg.DETECTION.ENABLE:
        # Validation for dn_models
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        dn_total_loss = val_dn_one_iter(all_preds, all_labels, dn_models)
        wandb.log({'dn val loss': dn_total_loss})
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
            cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    from datetime import datetime
    from utils import create_directory
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Initialize the Dependency Networks
    dn_models = get_model(cfg)
    logger.info("Loaded Dependency Networks")
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
            cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    # wandb.watch(model, log_freq=10)

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                        last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        dn_models = train_epoch(
                train_loader,
                model,
                dn_models,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                0,
                writer,
        )
        # for _ in range(cfg.JOINT_LEARNING.EXTRA_EPOCHS):
        #     dn_models = train_epoch_dn(train_loader, model, cfg, dn_models)
        epoch_timer.epoch_toc()
        logger.info(
                f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {start_epoch} to {cur_epoch} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
                f"For epoch {cur_epoch}, each iteraction takes "
                f"{epoch_timer.last_epoch_time() / len(train_loader):.2f}s in average. "
                f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time() / len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
                cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                    precise_bn_loader,
                    model,
                    min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                    cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch,
                    cfg,
                    scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            date_time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            model_save_location_torch = f'{cfg.OUTPUT_DIR}/dn_checkpoint/{date_time}/torch_{cur_epoch}/'
            create_directory(model_save_location_torch)
            model_directory = save_model(model_save_location_torch, dn_models)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, dn_models,
                       val_meter, cur_epoch, cfg, writer)

    # Save the dn_lr model

    from datetime import datetime
    date_time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    from utils import create_directory
    model_save_location_torch = f'{cfg.OUTPUT_DIR}/dn/{date_time}/torch/'

    create_directory(model_save_location_torch)
    model_directory = save_model(model_save_location_torch, dn_models)
    # for each_true_label, each_model in enumerate(dn_models):
    #     this_model_save_location_torch = model_save_location_torch + f"model_for_true_label_{each_true_label}"
    #     #https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    #     # Save the model in base form also
    #     # Saving torch model
    #     torch.save(each_model, this_model_save_location_torch)
    #     # Saving in model from scratch (LR)
    cfg.JOINT_LEARNING.MODEL_DIRECTORY = model_directory