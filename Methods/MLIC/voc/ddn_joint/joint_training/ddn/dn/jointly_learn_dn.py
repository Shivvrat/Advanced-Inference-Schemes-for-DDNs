import torch

from ddn_utils import save_data_to_pickle



def train_dn_one_iter(train_cnn_predictions_main, train_actual_output, dn_models, action="average"):
    """
    Train the DN and also returns the gradients wrt inputs to pass back to the CNN
    Args:
        dn_models:
        action:
        train_cnn_predictions_main:
        train_actual_output:

    Returns:

    """
    dn_models.train()
    # print(train_cnn_predictions_main.shape)
    # print(train_actual_output.shape)
    gradients, total_loss = dn_models.forward_train(train_cnn_predictions_main, train_actual_output,
                                                    action=action,
                                                    train=True, joint=True)
    # wandb.log({'dn loss': total_loss})

    return gradients, dn_models, total_loss


def val_dn_one_iter(inputs, labels, dn_models, action="average"):
    """
    Train the DN and also returns the gradients wrt inputs to pass back to the CNN
    Args:
        dn_models:
        action:
        cnn_predictions:
        labels:

    Returns:

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dn_models.eval()
    inputs = torch.cat(inputs, 0)
    labels = torch.cat(labels, 0)
    inputs = inputs.to(device)
    labels = labels.to(device)
    _, loss = dn_models.forward_train(inputs.float(), labels.float(), train=False, action=action)
    return loss



def save_model(model_save_location, this_model):
    this_model_save_location = f"{model_save_location}"
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    save_data_to_pickle(this_model, this_model_save_location)
    # torch.save(this_model, this_model_save_location)
    return this_model_save_location
