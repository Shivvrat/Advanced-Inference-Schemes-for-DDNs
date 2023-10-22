import os
from random import random

import numpy as np
import torch
from datetime import datetime

from natsort import natsorted
from numba import njit
from tqdm import tqdm
from torch.multiprocessing import Pool, Process, set_start_method


# from dn.inference.helper import get_model
from utils import get_data_from_slowfast_output_pkl, get_cuda_status_as_device, get_torch_float_tensor, \
    save_data_to_pickle, calculate_evaluation_metrics, create_directory, print_and_save_eval_metrics, init_data, \
    get_date_as_string

debug = None


# @njit
# def rand_choice_nb(arr, prob):
#     """
#     :param arr: A 1D numpy array of values to sample from.
#     :param prob: A 1D numpy array of probabilities for the given samples.
#     :return: A random sample from the given array with a given probability.
#     """
#     return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


# @njit
# def sample_from_probs_for_one_example(probabilities):
#     output = np.zeros_like(probabilities)
#     for i in range(probabilities.shape[0]):
#         for j in range(probabilities.shape[1]):
#             this_sample_prob = probabilities[i, j]
#             random_num = np.random.random()
#             if random_num <= this_sample_prob:
#                 this_sample_value = 1
#             else:
#                 this_sample_value = 0
#             # this_val = rand_choice_nb(np.array([0, 1]), prob=np.array([1 - this_sample_prob_1, this_sample_prob_1]))
#             output[i, j] = this_sample_value
#             # if debug and this_val > 0.3:
#             #     print("prob=" + str(this_sample_prob_1))
#             #     print("output=" + str(this_val))
#             #     print()
#     return output


@torch.no_grad()
def sample_one_instance(model, cnn_predictions, last_sample, var_sequence, device):
    this_sample = last_sample
    # input_for_model = torch.cat((this_sample, cnn_predictions), 1)
    this_sample, this_sample_prob_1 = model.forward_sampling(this_sample, cnn_predictions=cnn_predictions, var_sequence=var_sequence, device=device)
    this_sample_prob_1 = this_sample_prob_1.detach().cpu().numpy()
    # print(this_sample_prob_1)
    # this_sample_prob_1 will be an array with rows corresponding to examples and columns to labels
    # this_sample_value = np.random.choice([0, 1], p=[1 - this_sample_prob_1, this_sample_prob_1])
    # this_sample = sample_from_probs_for_one_example(this_sample_prob_1)
    # this_sample[this_true_label_index] = this_sample_value
    return this_sample_prob_1, this_sample


def sample(num_samples, num_true_labels, models, val_loader, device):
    var_sequence = np.arange(num_true_labels)
    outputs = []
    true_labels = []
    for cnn_predictions, labels in tqdm(val_loader):
        batch_size = cnn_predictions.shape[0]
        # todo Add the probs and divide by the number of samples at the end
        this_random_sample = np.random.binomial(n=1, p=0.5, size=[batch_size, num_true_labels])
        this_random_sample = torch.FloatTensor(this_random_sample).to(device)
        this_batch_samples_probs_sum = np.zeros((batch_size, num_true_labels))
        cnn_predictions = cnn_predictions.to(device)
        labels = labels.to(device)
        # this_batch_samples_probs_sum = this_random_sample
        for _ in range(num_samples):
            np.random.shuffle(var_sequence)
            # this_random_sample = torch.FloatTensor(this_random_sample).to(device)
            this_sample_prob_1, this_random_sample = sample_one_instance(models, cnn_predictions,
                                                                         this_random_sample, var_sequence, device)
            this_batch_samples_probs_sum += this_sample_prob_1
        this_sample_estimate = this_batch_samples_probs_sum / num_samples
        outputs.append(this_sample_estimate)
        true_labels.append(labels.detach().cpu().numpy())
    return outputs, true_labels


# @torch.no_grad()
# def gibbs_sampling(cfg, date, logger, models=None, params_string=None):
#     try:
#         set_start_method('spawn', force=True)
#         sharing_strategy = "file_system"
#         torch.multiprocessing.set_sharing_strategy(sharing_strategy)
#     except RuntimeError:
#         pass
#     slow_fast_directory_validation = cfg.DATASET.TEST_FILE_LOCATION
#     num_samples = cfg.SAMPLING.NUM_SAMPLES
#     global debug
#     debug = cfg.DEBUG
#     batch_size = cfg.TEST.BATCH_SIZE
#     device = get_cuda_status_as_device()
#     if models is None:
#         model_directory = cfg.MODEL.DIRECTORY
#         date = model_directory.split("/")[-2]
#         # device = torch.device("cpu")
#         models = get_model(model_directory, device)
#         params_string = model_directory.split("/")[-2]
#     # else:
#         # date = get_date_as_string()
#     models.eval()
#     valid_predictions, valid_actual_output = get_data_from_slowfast_output_pkl(slow_fast_directory_validation)
#     # for name, param in models.named_parameters():
#     #     if param.requires_grad:
#     #         print(name, param.data)
#     # exit()
#     if debug:
#         print(models)
#         num_examples_to_take = 100
#         valid_predictions, valid_actual_output = get_data_from_slowfast_output_pkl(cfg.DATASET.TRAIN_FILE_LOCATION)
#         valid_predictions = valid_predictions[:num_examples_to_take]
#         valid_actual_output = valid_actual_output[:num_examples_to_take]
#         batch_size = num_examples_to_take
#         num_samples = 100
#
#     num_examples, num_true_labels = valid_actual_output.shape
#     val_loader, this_valid_x, this_valid_y = init_data(valid_actual_output, valid_predictions, batch_size, device,
#                                                        train=False)
#     start = datetime.now()
#     outputs, true_labels = sample(num_samples, num_true_labels, models, val_loader, device)
#     outputs = np.vstack(outputs)
#     true_labels = np.vstack(true_labels)
#     if cfg.DEBUG:
#         print(outputs)
#     logger.info(f"Code run time - {datetime.now() - start}")
#     if debug:
#         output_path = f"{cfg.MODEL.SAVE_MODEL_PATH}debug/{params_string}/{date}/"
#     else:
#         output_path = f"{cfg.MODEL.SAVE_MODEL_PATH}{params_string}/{date}/"
#     if not debug:
#         output_data_location = f'{output_path}'
#         create_directory(output_data_location)
#         np.savetxt(f'{output_data_location}/test.output', outputs, delimiter=",")
#         for threshold in [0.1, 0.2, 0.3, 0.15, 0.25, 0.35]:
#             eval_metrics = calculate_evaluation_metrics(true_labels, outputs, threshold)
#             results_dir = os.path.join(output_path, f'results/')
#             create_directory(results_dir)
#             output_filename = f'{results_dir}threshold_{threshold}_results.csv'
#             print_and_save_eval_metrics(eval_metrics, output_filename, logger)
#             # print(eval_metrics)
