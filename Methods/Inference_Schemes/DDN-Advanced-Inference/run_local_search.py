import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import wandb
from model_class.dn_lr import LR
from model_class.dn_nn import NN
from utils import (
    get_cnn_output_and_model,
    init_logger_and_wandb,
    calculate_evaluation_metrics,
    print_and_save_eval_metrics,
    save_metrics_and_sample_n_iter,
)
from search_methods import random_walk, greedy_search


def local_search(
    num_samples,
    device,
    load_existing_samples,
    dataset_name,
    dn_type,
    SAMPLES_FILE,
    model,
    num_true_labels,
    loader,
):
    sampling_outputs = []
    all_labels = []
    with torch.no_grad():
        for cnn_predictions, labels in tqdm(loader):
            cnn_predictions = cnn_predictions.to(device)
            batch_size = cnn_predictions.shape[0]
            labels = labels.to(device)
            var_sequence = np.random.permutation(np.arange(num_true_labels))
            if not load_existing_samples:
                this_random_sample = copy.deepcopy(cnn_predictions).to(device)
            this_random_sample[this_random_sample >= 0.3] = 1
            this_random_sample[this_random_sample < 0.3] = 0
            best_samples = copy.deepcopy(this_random_sample)
            cnn_predictions = cnn_predictions.to(device)
            labels = labels.to(device)
            previous_best_pll_scores = None
            no_improvement_counter = 0

            for sample_idx in range(num_samples):
                if args.search_method == "greedy":
                    this_random_sample, current_best_pll_scores = greedy_search(
                        device,
                        model,
                        cnn_predictions,
                        batch_size,
                        var_sequence,
                        this_random_sample,
                    )
                elif args.search_method == "random":
                    this_random_sample, current_best_pll_scores = random_walk(
                        device,
                        model,
                        cnn_predictions,
                        batch_size,
                        var_sequence,
                        this_random_sample,
                    )

                # Check for improvement
                if previous_best_pll_scores is not None:
                    if torch.all(current_best_pll_scores <= previous_best_pll_scores):
                        no_improvement_counter += 1
                    else:
                        no_improvement_counter = (
                            0  # Reset counter if there's an improvement
                        )
                previous_best_pll_scores = current_best_pll_scores

                # Terminate loop if no improvement for 100 consecutive iterations
                if no_improvement_counter >= MAX_NO_IMPROVEMENT_ITERATIONS:
                    break
                #     best_samples = this_random_sample
                #     best_ll_score = current_best_pll_scores
                # else:
                #     this_random_sample = torch.randint_like(cnn_predictions, 2)

                if sample_idx % 500 == 0:
                    save_metrics_and_sample_n_iter(
                        dataset_name,
                        dn_type,
                        SEARCH_METHOD,
                        SAMPLES_FILE,
                        labels,
                        this_random_sample,
                        sample_idx,
                    )
            this_random_sample = this_random_sample.cpu().numpy()

            labels = labels.cpu().numpy()
            sampling_outputs.append(this_random_sample)
            all_labels.append(labels)
    sampling_outputs = np.concatenate(sampling_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = calculate_evaluation_metrics(all_labels, sampling_outputs, 0.3)
    directory = f"metrics/{SEARCH_METHOD}/last_iter"
    os.makedirs(directory, mode=0o777, exist_ok=True)
    print_and_save_eval_metrics(
        metrics,
        f"{directory}/{dataset_name}_{args.dn_type}.csv",
    )
    np.save(SAMPLES_FILE.replace(".pt", "_final.npy"), sampling_outputs)


NUM_SAMPLES = 5000
MAX_NO_IMPROVEMENT_ITERATIONS = 1000

logger.info(f"num_samples: {NUM_SAMPLES}")
os.makedirs("metrics", mode=0o777, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="voc", help="name of dataset")
parser.add_argument("--dn_type", type=str, default="lr", help="name of dn_type")
parser.add_argument(
    "--search_method",
    type=str,
    default="greedy",
    help="name of method you want to use",
    choices=[
        "random",
        "greedy",
    ],
)

logger.info("Parsing arguments")
args = parser.parse_args()
logger.info(args)
project_name = f"{args.search_method}_{args.dataset}_{args.dn_type}"
init_logger_and_wandb(project_name, args)
load_existing_samples = False

DATASET_NAME = args.dataset
DN_TYPE = args.dn_type
SEARCH_METHOD = args.search_method
SAMPLES_FILE = f"samples/{args.search_method}/{DATASET_NAME}_{DN_TYPE}.pt"
os.makedirs(os.path.dirname(SAMPLES_FILE), exist_ok=True)
# Check for existing samples
if os.path.exists(SAMPLES_FILE) and load_existing_samples:
    this_random_sample = torch.load(SAMPLES_FILE).to(device)
logger.info(
    f"dataset: {DATASET_NAME}, dn_type: {DN_TYPE}, sampling_method: {SEARCH_METHOD}"
)
# for dataset in ["nus", "voc", "charades", "coco"]:
# for dn_type in ["nn", "lr"]:
cnn_outputs, true_labels, model = get_cnn_output_and_model(DATASET_NAME, DN_TYPE)
BATCH_SIZE = cnn_outputs.shape[0]

model.to()
num_true_labels = true_labels.shape[1]
dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(cnn_outputs), torch.FloatTensor(true_labels)
)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
local_search(
    NUM_SAMPLES,
    device,
    load_existing_samples,
    DATASET_NAME,
    DN_TYPE,
    SAMPLES_FILE,
    model,
    num_true_labels,
    loader,
)
