import torch

from utils import save_data_to_pickle
import wandb


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
    gradients, total_loss = dn_models.forward_train(train_cnn_predictions_main, train_actual_output,
                                                    action=action,
                                                    train=True)
    # wandb.log({'dn loss': total_loss})

    return gradients, dn_models, total_loss
    #
    #
    # Everything done inside the dn class
    #
    # for each_true_label_index in range(num_true_label):
    #     this_train_y = train_actual_output[:, each_true_label_index]
    #     if each_true_label_index == 0:
    #         all_other_actual_labels = train_actual_output[:, 1:]
    #     elif each_true_label_index == num_true_label - 1:
    #         all_other_actual_labels = train_actual_output[:, :-1]
    #     else:
    #         all_other_actual_labels = torch.cat((train_actual_output[:, :each_true_label_index],
    #                                              train_actual_output[:, each_true_label_index + 1:]), 1)
    #     # Same train_cnn_predictions used for each iter - We need to make a clone of the one given in the function
    #     train_cnn_predictions = train_cnn_predictions_main.detach().clone()
    #     train_cnn_predictions.requires_grad = True
    #     this_train_x = torch.cat((all_other_actual_labels, train_cnn_predictions), 1)
    #     # this_train_x.requires_grad = True
    #     # this_train_y = get_torch_float_tensor(this_train_y, device)
    #     # this_train_x = get_torch_float_tensor(this_train_x, device)
    #     optimizers[each_true_label_index].zero_grad()
    #     # outputs = all_models[each_true_label_index](this_train_x.float())
    #     outputs = all_models[each_true_label_index](this_train_x)
    #     final_outputs.append(outputs)
    #     loss = criterion(torch.squeeze(outputs), this_train_y)
    #     # loss = criterion(torch.squeeze(outputs), this_train_y.float())
    #     # Compute L1 loss component
    #     l1_parameters = [parameter.view(-1) for parameter in all_models[each_true_label_index].parameters()]
    #     l1 = L1_WEIGHT * all_models[each_true_label_index].compute_l1_loss(torch.cat(l1_parameters))
    #     # Add L1 loss component
    #     loss += l1
    #     loss.backward()
    #     gradient_wrt_train_cnn_predictions = train_cnn_predictions.grad
    #     gradients_for_CNN.append(gradient_wrt_train_cnn_predictions)
    #     optimizers[each_true_label_index].step()
    # tensor = torch.stack(gradients_for_CNN)
    # if action.strip().lower() == "average":
    #     return torch.mean(tensor, dim=0), all_models
    # elif action.strip().lower() == "sum":
    #     return torch.sum(tensor, dim=0), all_models


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


# def train_epoch_dn(train_loader, cnn_model, cfg, dn_models):
#     """
#     Perform the video training for one epoch.
#     Args:
#         train_loader (loader): video training loader.
#         cnn_model (model): the video model to train.
#         cfg (CfgNode): configs. Details can be found in
#             slowfast/config/defaults.py
#     """
#     # Enable train mode.
#     cnn_model.eval()
#     dn_models.train()
#     for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
#         # Transfer the data to the current GPU device.
#         if cfg.NUM_GPUS:
#             if isinstance(inputs, (list,)):
#                 for i in range(len(inputs)):
#                     inputs[i] = inputs[i].cuda(non_blocking=True)
#             else:
#                 inputs = inputs.cuda(non_blocking=True)
#             labels = labels.cuda()
#             for key, val in meta.items():
#                 if isinstance(val, (list,)):
#                     for i in range(len(val)):
#                         val[i] = val[i].cuda(non_blocking=True)
#                 else:
#                     meta[key] = val.cuda(non_blocking=True)
#
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
#                 if cfg.DETECTION.ENABLE:
#                     preds = cnn_model(inputs, meta["boxes"])
#                 else:
#                     preds = cnn_model(inputs)
#                 # Explicitly declare reduction to mean.
#                 # Compute the loss.
#         # Get gradients from dn_lr
#         _, dn_models = train_dn_one_iter(preds.detach().float(), labels.float(), dn_models, action="average")
#     return dn_models


def save_model(model_save_location, this_model):
    this_model_save_location = f"{model_save_location}trained_dn_nn_model"
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    save_data_to_pickle(this_model, this_model_save_location)
    # torch.save(this_model, this_model_save_location)
    return this_model_save_location
