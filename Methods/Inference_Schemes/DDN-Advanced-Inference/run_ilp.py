import os
import argparse
import torch
from loguru import logger
from ilp.ilp_runner import run_ilp
from ilp.load_model import convert_dn_numpy_dict
from utils import (
    get_cnn_output_and_model,
    init_logger_and_wandb,
)

# Constants and Configurations
os.makedirs("metrics", mode=0o777, exist_ok=True)
SAMPLES_DIR = "samples/ILP"
os.makedirs(SAMPLES_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nus", help="name of dataset")
    parser.add_argument("--dn_type", type=str, default="lr", help="name of dn_type")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--batch_size", type=int, default=64, help="number of pools")
    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info(args)
    logger.info(f"Debug mode: {args.debug}")

    project_name = f"ILP_{args.dataset}_{args.dn_type}"
    init_logger_and_wandb(project_name, args)

    SAMPLES_FILE = os.path.join(SAMPLES_DIR, f"{args.dataset}_{args.dn_type}.pt")
    cnn_outputs, true_labels, model = get_cnn_output_and_model(
        args.dataset, args.dn_type
    )
    ddn_as_dict = convert_dn_numpy_dict(model)
    logger.success("Converted model to numpy dict")

    run_ilp(ddn_as_dict, args, cnn_outputs, true_labels, SAMPLES_FILE, debug=False)


if __name__ == "__main__":
    main()
