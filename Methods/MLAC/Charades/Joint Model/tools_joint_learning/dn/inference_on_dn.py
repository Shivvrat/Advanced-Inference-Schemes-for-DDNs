import copy
import pickle
import os
import sys

import numpy as np
import torch
import wandb
from torch.multiprocessing import Pool, Process, set_start_method

from utils import get_data_from_slowfast_output_pkl, get_cuda_status_as_device, get_torch_float_tensor, \
    save_data_to_pickle, calculate_evaluation_metrics, create_directory, print_and_save_eval_metrics, init_data, \
    get_date_as_string

from dn.inference.gibbs_sampling import sample

debug = None


def get_model(cfg):
    """
    Import the models from given directory
    :param model_directory: the directory where models are stored
    :param device: the device to which we want to use for evaluation
    :return: list of models according to the index

    Args:
        cfg:
    """
    with open(cfg.JOINT_LEARNING.MODEL_DIRECTORY, 'rb') as file:
        if cfg.JOINT_LEARNING.DN_TYPE == "nn":
            from model_class.dn_nn import DependencyNetwork
        elif cfg.JOINT_LEARNING.DN_TYPE == "lr":
            from model_class.dn_lr import DependencyNetwork
        else:
            raise TypeError("Incorrect Model Type given")
        this_model = pickle.load(file)
    this_model.to()

    return this_model


@torch.no_grad()
def gibbs_sampling(valid_actual_output, valid_predictions, models, num_samples, model_save_path, batch_size, logger):
    num_examples, num_true_labels = valid_actual_output.shape
    try:
        set_start_method('spawn', force=True)
        sharing_strategy = "file_system"
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    except RuntimeError:
        pass
    device = get_cuda_status_as_device()
    # Start eval mode for NNs
    models.eval()
    # for name, param in models.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    # exit()
    val_loader, this_valid_x, this_valid_y = init_data(valid_actual_output, valid_predictions, batch_size, device,
                                                       train=False)
    outputs, true_labels = sample(num_samples, num_true_labels, models, val_loader, device)
    outputs = np.vstack(outputs)
    true_labels = np.vstack(true_labels)
    if not debug:
        probs = copy.deepcopy(outputs)
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            eval_metrics = calculate_evaluation_metrics(true_labels, outputs, threshold)
            results_dir = f'{model_save_path}/dn_results/'
            create_directory(results_dir)
            output_filename = f'{results_dir}threshold_{threshold}_results.csv'
            print(eval_metrics)
            if threshold == 0.3:
                wandb.alert(title="Evaluation metrics",
                            text=f"{eval_metrics}")
            print_and_save_eval_metrics(eval_metrics, output_filename, logger)
        output_path = f'{model_save_path}/dn_outputs/'
        output_data_location = f'{output_path}/threshold_{threshold}'
        create_directory(output_data_location)
        np.savetxt(f'{output_data_location}/test.output', probs, delimiter=",")
