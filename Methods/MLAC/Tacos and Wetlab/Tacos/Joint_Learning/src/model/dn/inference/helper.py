import pickle

import torch


def get_model(model_directory, device):
    """
    Import the models from given directory
    :param model_directory: the directory where models are stored
    :param device: the device to which we want to use for evaluation
    :return: list of models according to the index
    """
    from  model_class.dn import DependencyNetwork
    # if device == torch.device("cpu"):
    #     this_model = torch.load(model_directory)
    # else:
    #     this_model = torch.load(model_directory, map_location=torch.device('cpu'))
    with open(model_directory, 'rb') as file:
        this_model = pickle.load(file)
    this_model.to(device)
    return this_model