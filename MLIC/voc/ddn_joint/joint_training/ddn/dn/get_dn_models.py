import pickle

# from dn_lr.model_class.logistic_regression_model import LogisticRegressionModel
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD


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
        if cfg.MODEL.DN_TYPE == "nn":
            from model_class.dn_nn import DependencyNetwork
        elif cfg.MODEL.DN_TYPE == "lr":
            from model_class.dn_lr import DependencyNetwork
        else:
            raise TypeError("Incorrect ddn model type given")
        with open(cfg.JOINT_LEARNING.MODEL_DIRECTORY, 'rb') as file:
            this_model = pickle.load(file)
        this_model.to()
        this_model.criterion = criterion
        this_model.optimizers = [SGD(each_model.parameters(), lr=cfg.JOINT_LEARNING.LEARNING_RATE,
                                     weight_decay=cfg.JOINT_LEARNING.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM,)
         for each_model in this_model.nns]
        this_model.cfg = cfg
    else:
        print("We are using a new model")
        if cfg.MODEL.DN_TYPE == "nn":
            from model_class.dn_nn import DependencyNetwork
        elif cfg.MODEL.DN_TYPE == "lr":
            from model_class.dn_lr import DependencyNetwork
        num_classes = cfg.MODEL.NUM_CLASSES
        this_model = DependencyNetwork(num_classes*2, num_classes, device, criterion, cfg)
    return this_model