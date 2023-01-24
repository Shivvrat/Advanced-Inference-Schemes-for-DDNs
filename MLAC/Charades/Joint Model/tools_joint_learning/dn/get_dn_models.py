import pickle
import sys
from glob import glob

from natsort import natsorted
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import SGD, Adam

# from dn_lr.model_class.logistic_regression_model import LogisticRegressionModel
from utils import import_from_pickle_file
import torch


def get_cuda_status_as_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_model(cfg):
    """
    Import the models from given directory
    :param model_directory: the directory where models are stored
    :param device: the device to which we want to use for evaluation
    :return: list of models according to the index
    """
    device = get_cuda_status_as_device()
    criterion = BCEWithLogitsLoss()
    if cfg.JOINT_LEARNING.PRETRAINED:
        print("We are using a pre-trained model")
        with open(cfg.JOINT_LEARNING.MODEL_DIRECTORY, 'rb') as file:
            if cfg.JOINT_LEARNING.DN_TYPE == "nn":
                from model_class.dn_nn import DependencyNetwork
            elif cfg.JOINT_LEARNING.DN_TYPE == "lr":
                from model_class.dn_lr import DependencyNetwork
            else:
                raise TypeError("Incorrect Model Type given")
            this_model = pickle.load(file)
        this_model.to()
        this_model.criterion = criterion
        this_model.optimizers = [SGD(each_model.parameters(), lr=cfg.JOINT_LEARNING.LEARNING_RATE,
                                     weight_decay=cfg.JOINT_LEARNING.WEIGHT_DECAY)
                                 for each_model in this_model.nns]
    else:
        print("We are using a new model")
        if cfg.JOINT_LEARNING.DN_TYPE == "nn":
            from model_class.dn_nn import DependencyNetwork
        elif cfg.JOINT_LEARNING.DN_TYPE == "lr":
            from model_class.dn_lr import DependencyNetwork
        else:
            raise TypeError("Incorrect Model Type given")
        num_classes = cfg.MODEL.NUM_CLASSES
        this_model = DependencyNetwork(num_classes * 2, num_classes, device, criterion, cfg)
    return this_model
