def model_to_numpy_dict(model):
    """
    Convert all the parameters in the neural network to numpy arrays and store them in a dictionary.
    :param model: Trained neural network model
    :return: Dictionary of model parameters as numpy arrays and the original model
    """
    nn_as_dict_of_np_array = {}
    for param_tensor in model.state_dict():
        nn_as_dict_of_np_array[param_tensor] = (
            model.state_dict()[param_tensor].cpu().detach().numpy()
        )
    return nn_as_dict_of_np_array


def load_layer_info(nn_as_dict):
    """
    Extract layer-wise information from a dictionary of model parameters.

    :param nn_as_dict: Dictionary of neural network parameters as numpy arrays
    :return: Layer-wise structured information
    """
    layer_info = {}
    for key, value in nn_as_dict.items():
        layer, param_type = key.rsplit(".", 1)
        if layer not in layer_info:
            layer_info[layer] = {}
        layer_info[layer][param_type] = value

    idx = 0
    final_layer_info = {}
    for key, value in layer_info.items():
        final_layer_info[idx] = value
        idx += 1

    return final_layer_info


def convert_dn_numpy_dict(dn):
    """
    Convert a list of neural networks to a dictionary of numpy arrays, organized layer-wise.

    :param dn: List of neural networks
    :return: Dictionary containing layer-wise structured information for each neural network
    """
    ddn_dict = {}

    for idx in range(len(dn.nns)):
        nn_as_dict = load_layer_info(model_to_numpy_dict(dn.nns[idx]))
        ddn_dict[idx] = nn_as_dict

    return ddn_dict
