import copy
import pickle
import os
import sys

import numpy as np
import torch
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
    """
    with open(cfg.MODEL.SAVE_MODEL_PATH + "trained_dn_nn_model", 'rb') as file:
        from model_class.dn_nn import DependencyNetwork
        this_model = pickle.load(file)
    this_model.to()

    return this_model


@torch.no_grad()
def gibbs_sampling(valid_actual_output, valid_predictions, models, num_samples, model_save_path, batch_size, logger,
                   cfg):
    num_examples, num_true_labels = valid_actual_output.shape
    global debug
    debug = cfg.DEBUG
    if not cfg.DEBUG:
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            eval_metrics = calculate_evaluation_metrics(valid_actual_output, valid_predictions, threshold)
            results_dir = f'{model_save_path}/cnn_results/'
            create_directory(results_dir)
            output_filename = f'{results_dir}threshold_{threshold}_results.csv'
            print(eval_metrics)
            print_and_save_eval_metrics(eval_metrics, output_filename, logger)
    try:
        set_start_method('spawn', force=True)
        sharing_strategy = "file_system"
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    except RuntimeError:
        pass
    device = get_cuda_status_as_device()
    # Start eval mode for NNs
    models.eval()
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
            print_and_save_eval_metrics(eval_metrics, output_filename, logger)
        output_path = f'{model_save_path}/'
        create_directory(output_path)
        np.savetxt(f'{output_path}/test_dn.outputs', probs, delimiter=",")
