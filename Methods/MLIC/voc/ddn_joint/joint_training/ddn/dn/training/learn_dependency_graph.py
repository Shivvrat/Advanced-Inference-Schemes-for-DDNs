import cProfile
import io
import pstats
from pstats import SortKey

import math
import torch
import wandb
from ddn_utils import create_directory, get_cuda_status_as_device, get_data_from_slowfast_output_pkl, \
    get_date_as_string, get_loss_and_accuracy_pytorch, init_data, is_eval_epoch, plot_losses, save_data_to_pickle
from model_class.dn import DependencyNetwork
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

debug = False


def train_epoch(train_loader, device, this_model):
    this_model.train()
    for iter, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # this_model.zero_grad()
        # optimizer.zero_grad()
        outputs, _ = this_model.forward_train(inputs.float(), labels.float(), train=True)
        outputs = outputs.to(device)
        # loss = criterion(torch.squeeze(outputs), labels.float())
        if debug:
            print("max prob=" + str(torch.max(torch.sigmoid(outputs), dim=1)))
            print("true label=" + str(torch.sum(labels, dim=1)))
            print()
        # loss = criterion(outputs, labels.float())
        # if l1:
        #     # Compute L1 loss component
        #     l1_parameters = []
        #     for parameter in this_model.parameters():
        #         l1_parameters.append(parameter.view(-1))
        #     l1 = l1_weight * this_model.compute_l1_loss(torch.cat(l1_parameters))
        #     # Add L1 loss component
        #     loss += l1
        # loss.backward()
        # optimizer.step()


@torch.no_grad()
def val_epoch(this_train_x, this_train_y, this_valid_x, this_valid_y, this_model, criterion, logger, last_loss, epoch,
              epochs, device):
    this_model.eval()
    train_accuracy, train_loss = get_loss_and_accuracy_pytorch(this_train_x, this_train_y, this_model, criterion,
                                                               device)
    test_accuracy, test_loss = get_loss_and_accuracy_pytorch(this_valid_x, this_valid_y, this_model, criterion, device)
    stop_train = False
    if abs(last_loss - train_loss) < 1e-3:
        logger.info("Early Stopping")
        logger.info(f"Epoch: {epoch}. Test - Loss: {test_loss}. Accuracy: {test_accuracy}")
        logger.info(f"Train -  Loss: {train_loss}. Accuracy: {train_accuracy}")
        logger.info(f"Last -  Loss: {last_loss}")
        stop_train = True
    last_loss = train_loss
    if epoch + 1 % (epochs / 50) == 0:
        logger.info(f"Epoch: {epoch}. \nTest - Loss: {test_loss}. Accuracy: {test_accuracy}")
        logger.info(f"Train -  Loss: {train_loss}. Accuracy: {train_accuracy}")
        # plot_losses(loss_per_iter)
    return train_loss, test_loss, last_loss, stop_train, train_accuracy, test_accuracy


def save_model(model_save_location, this_model):

    this_model_save_location = f"{model_save_location}trained_dn_nn_model"
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    save_data_to_pickle(this_model, this_model_save_location)
    # torch.save(this_model, this_model_save_location)


def learn_dn(cfg, logger):
    device = get_cuda_status_as_device()
    # Get train data
    slow_fast_directory_train = cfg.DATASET.TRAIN_FILE_LOCATION
    train_predictions, train_actual_output = get_data_from_slowfast_output_pkl(slow_fast_directory_train)
    # Get validation data
    slow_fast_directory_validation = cfg.DATASET.TEST_FILE_LOCATION
    valid_predictions, valid_actual_output = get_data_from_slowfast_output_pkl(slow_fast_directory_validation)
    date = get_date_as_string()
    global debug
    debug = cfg.DEBUG
    # Load model hyper parameters
    epochs = cfg.MODEL.EPOCHS
    input_dim = train_predictions.shape[1] * 2
    output_dim = train_predictions.shape[1]
    learning_rate = cfg.MODEL.LEARNING_RATE
    batch_size = cfg.TRAIN.BATCH_SIZE
    # l1_weight = cfg.MODEL.L1_WEIGHT
    l2_weight = cfg.SOLVER.WEIGHT_DECAY
    if debug:
        params_string = f"debug/epochs_{epochs}_lr_{learning_rate}_l2_weight_{l2_weight}"

    else:
        params_string = f"epochs_{epochs}_lr_{learning_rate}_l2_weight_{l2_weight}"

    model_save_location = f"{cfg.MODEL.SAVE_MODEL_PATH}{date}/{params_string}/"
    create_directory(model_save_location)
    # Info - For x we take actual labels first and then predictions from CNN
    criterion = BCEWithLogitsLoss()
    if debug:
        # Debug mode?
        logger.info("We are in Debug Model")
        train_actual_output = train_actual_output[:10]
        train_predictions = train_predictions[:10]
        valid_actual_output = valid_actual_output[:5]
        valid_predictions = valid_predictions[:5]
        batch_size = 5
    train_loader, this_train_x, this_train_y = init_data(train_actual_output, train_predictions, batch_size, device,
                                                         train=True)
    val_loader, this_valid_x, this_valid_y = init_data(valid_actual_output, valid_predictions, batch_size, device,
                                                       train=False)
    this_model = DependencyNetwork(input_dim, output_dim, device, criterion, cfg)

    # optimizer = Adam(this_model.parameters(), lr=learning_rate, weight_decay=l2_weight)
    losses_test = []
    losses_train = []
    last_loss = math.inf
    # Train the model
    for epoch in tqdm(range(epochs)):
        train_epoch(train_loader, device, this_model)
        if debug:
            pr = cProfile.Profile()
            pr.enable()
        # Function contents
        if is_eval_epoch(cfg, epoch):
            train_loss, test_loss, last_loss, stop_train, train_accuracy, test_accuracy = val_epoch(this_train_x,
                                                                                                    this_train_y,
                                                                                                    this_valid_x,
                                                                                                    this_valid_y,
                                                                                                    this_model,
                                                                                                    criterion,
                                                                                                    logger, last_loss,
                                                                                                    epoch,
                                                                                                    epochs, device)
        if debug:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.TIME
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
        if not debug:
            wandb.log({f"train_loss_{params_string}": train_loss}, step=epoch)
            # wandb.log({f"test_loss_{params_string}": test_loss}, step=epoch)
            # wandb.log({f"train_accuracy_{params_string}": train_accuracy}, step=epoch)
            # wandb.log({f"test_accuracy_{params_string}": test_accuracy}, step=epoch)
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        # Stop if early stopping criterion is true
        if stop_train:
            logger.info("Early Stopping")
            break

    plot_losses(losses_test, params_string, losses_train)
    if not debug:
        save_model(model_save_location, this_model)
    return this_model
