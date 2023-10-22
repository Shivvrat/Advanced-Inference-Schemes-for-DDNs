import multiprocessing
from multiprocessing import Pool
import os

import numpy as np
from loguru import logger
from tqdm import tqdm
from ilp.optimize_mpe import compute_mpe_using_gurobi

from utils import (
    calculate_evaluation_metrics,
    print_and_save_eval_metrics,
)


def get_2d_np_array(data):
    values = [np.array(xi).reshape((-1)) for xi in data if np.array(xi).size > 0]
    return np.vstack(values)


import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger
import os
import cProfile


from tqdm import tqdm


def run_ilp(dn_as_dict, args, cnn_features, target, SAMPLES_FILE, debug=False):
    all_mpe_outputs = []
    all_targets = []
    missed_examples = []

    num_pools = args.batch_size if debug else (multiprocessing.cpu_count()) * 4 // 5
    if debug:
        num_pools = 1

    logger.info(f"We will use {num_pools} Pools")
    logger.info("Getting MPE values DN using ILP")

    with Pool(processes=num_pools) as pool:
        if debug:
            cnn_features = cnn_features[:2]
            target = target[:2]

            pr = cProfile.Profile()
            pr.enable()

        inputs_gen = (
            (dn_as_dict, cnn_features[i], target[i], i)
            for i in range(len(cnn_features))
        )

        ilp_outputs = list(
            tqdm(
                pool.imap(compute_mpe_using_gurobi, inputs_gen), total=len(cnn_features)
            )
        )

        for idx, mpe_output, true in ilp_outputs:
            if mpe_output is not None:
                all_mpe_outputs.append(mpe_output)
                all_targets.append(true)
            else:
                missed_examples.append(idx)

        if debug:
            pr.disable()
            logger.info(f"Initial Inputs {all_targets}")
            logger.info(f"Updated Inputs {all_mpe_outputs}")
            metrics = calculate_evaluation_metrics(all_targets, all_mpe_outputs, 0.3)
            print(metrics)

    if not debug:
        all_mpe_outputs = get_2d_np_array(all_mpe_outputs)
        all_targets = get_2d_np_array(all_targets)

        np.savez(
            SAMPLES_FILE.replace(".pt", "_final.npy"),
            mpe_outputs=all_mpe_outputs,
            initial=all_targets,
            missed_examples=np.array(missed_examples),
        )
        metrics = calculate_evaluation_metrics(all_targets, all_mpe_outputs, 0.3)
        directory = "metrics/ILP/"
        os.makedirs(directory, mode=0o777, exist_ok=True)
        print_and_save_eval_metrics(
            metrics, f"{directory}/{args.dataset}_{args.dn_type}.csv"
        )

    return all_mpe_outputs, all_targets, missed_examples
