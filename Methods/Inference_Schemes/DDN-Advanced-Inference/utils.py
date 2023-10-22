import os
import pickle
from loguru import logger

import numpy as np
import torch
import wandb
from model_class.dn_lr import LR
from model_class.dn_nn import NN


def get_cnn_output_and_model(dataset, method):
    print(f"dataset: {dataset}, method: {method}")
    print(dataset, method)
    base_path = "/data/DDN/data/"
    if not os.path.exists(base_path):
        raise ValueError("No data path found")
    if dataset == "charades":
        model_path = f"{base_path}charades/{method}/trained_dn_{method}_model"
        cnn_output_path = f"{base_path}charades/{method}/val.pkl"
        cnn_outputs = pickle.load(open(cnn_output_path, "rb"))
        cnn_outputs, true_labels = cnn_outputs
        cnn_outputs, true_labels = cnn_outputs.numpy(), true_labels.numpy()

    elif dataset in ["coco", "nus", "voc"]:
        model_path = f"{base_path}{dataset}/{method}/dn_{method}.pkl"
        if dataset == "coco":
            cnn_output_path = f"{base_path}{dataset}/{method}/val_saved_data.0.txt"
            cnn_outputs = np.genfromtxt(cnn_output_path, delimiter=" ")

            # cnn_outputs = open(cnn_output_path, "r").readlines()

        elif dataset in ["nus", "voc"]:
            cnn_output_path = f"{base_path}{dataset}/{method}/val_outputs.csv"
            # open csv as numpy array
            cnn_outputs = np.loadtxt(cnn_output_path, delimiter=" ", dtype=float)
        cnn_outputs, true_labels = (
            cnn_outputs[:, : cnn_outputs.shape[1] // 2],
            cnn_outputs[:, cnn_outputs.shape[1] // 2 :],
        )
    elif dataset in ["wetlab", "tacos"]:
        model_path = f"{base_path}{dataset}/{method}/trained_dn_{method}_model"
        cnn_output_path = f"{base_path}{dataset}/{method}/cnn_outputs.pkl"
        cnn_outputs = pickle.load(open(cnn_output_path, "rb"))
        cnn_outputs, true_labels = (
            cnn_outputs["joint_learning_cnn_with_mn"],
            cnn_outputs["ground_truth"],
        )
    with open(model_path, "rb") as file:
        this_model = pickle.load(file)
    print(this_model)
    print(cnn_outputs.shape)
    print(true_labels.shape)
    return cnn_outputs, true_labels, this_model


import numpy as np
from sklearn.metrics import (
    hamming_loss,
    label_ranking_loss,
    coverage_error,
    label_ranking_average_precision_score,
    accuracy_score,
    average_precision_score,
    jaccard_score,
)
import pandas as pd


def calculate_evaluation_metrics(y_true, y_pred_probability, threshold):
    y_pred = np.copy(y_pred_probability)
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0

    metrics = {
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Coverage": coverage_error(y_true, y_pred_probability),
        "Ranking Loss": label_ranking_loss(y_true, y_pred_probability),
        "Average Precision": average_precision_score(y_true, y_pred_probability),
        "Subset Accuracy": accuracy_score(y_true, y_pred),
        "Jaccard Score": jaccard_score(y_true, y_pred, average="samples"),
        "Label Ranking Average Precision Score": label_ranking_average_precision_score(
            y_true, y_pred_probability
        ),
    }

    return metrics


def print_and_save_eval_metrics(metrics, output_filename, iteration_number=None):
    # Display the metrics
    for metric_name, metric_value in metrics.items():
        if iteration_number is not None:
            print(f"Iteration {iteration_number} - {metric_name} : {metric_value}")
        else:
            print(f"{metric_name} : {metric_value}")

    # Determine the column name based on iteration number and timestamp
    if iteration_number is not None:
        column_name = f"Iteration {iteration_number} - {pd.Timestamp.now()}"
    else:
        column_name = "Initial"

    # Check if the file exists
    if os.path.exists(output_filename):
        # Load the existing DataFrame
        existing_df = pd.read_csv(output_filename, index_col=0)
        # Append the new data as a column
        existing_df[column_name] = pd.Series(metrics)
        # Save the updated DataFrame
        existing_df.to_csv(output_filename)
    else:
        # Create a new DataFrame and save it
        df = pd.DataFrame.from_dict(data=metrics, orient="index", columns=[column_name])
        df.to_csv(output_filename)


def get_best_metrics(output_filename, metric_name, maximize=True):
    # Load the DataFrame
    df = pd.read_csv(output_filename, index_col=0)

    # Ensure the metric exists
    if metric_name not in df.index:
        raise ValueError(f"Metric '{metric_name}' not found in the file.")

    # Get the column (iteration) with the best value for the specified metric
    if maximize:
        best_iteration = df.loc[metric_name].idxmax()
        best_value = df.loc[metric_name].max()
    else:
        best_iteration = df.loc[metric_name].idxmin()
        best_value = df.loc[metric_name].min()

    return best_iteration, best_value


def init_logger_and_wandb(project_name, args):
    import sys

    wandb.init(project=project_name)
    wandb.config.update(args)
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
            {
                "sink": "logging/" + "logger_{time}.log",
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
        ],
        "extra": {"user": "usr"},
    }
    logger.configure(**config)
    return


def save_metrics_and_sample_n_iter(
    dataset_name,
    method,
    search_method,
    SAMPLES_FILE,
    labels,
    this_random_sample,
    sample_idx,
):
    this_random_sample_cpu = this_random_sample.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    torch.save(this_random_sample_cpu, SAMPLES_FILE)
    metrics = calculate_evaluation_metrics(labels_cpu, this_random_sample_cpu, 0.3)
    wandb.log(metrics)
    directory = f"metrics/{search_method}"
    os.makedirs(directory, mode=0o777, exist_ok=True)
    print_and_save_eval_metrics(
        metrics,
        f"{directory}/{dataset_name}_{method}.csv",
        iteration_number=sample_idx,
    )


import torch


def flip_some_elements(tensor, percent=0.1):
    """
    Randomly flip 30% of the values in a 2D binary tensor.

    Parameters:
    - tensor: torch.Tensor
        A 2D binary tensor with values 0 or 1.

    Returns:
    - torch.Tensor
        The modified tensor with 10% values flipped.
    """

    # Ensure the tensor is 2D
    assert len(tensor.shape) == 2, "Tensor must be 2D"

    # Calculate the number of elements to flip
    total_elements = tensor.numel()
    num_to_flip = int(percent * total_elements)

    # Randomly select indices to flip
    flip_indices = torch.multinomial(
        torch.ones(total_elements), num_to_flip, replacement=False
    )

    # Convert flat indices to 2D indices
    rows = flip_indices // tensor.shape[1]
    cols = flip_indices % tensor.shape[1]

    # Flip the values
    tensor[rows, cols] = 1 - tensor[rows, cols]

    return tensor
