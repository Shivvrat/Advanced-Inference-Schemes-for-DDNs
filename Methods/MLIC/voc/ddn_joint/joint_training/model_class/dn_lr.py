import numpy as np
import torch
from numba import njit
from torch import nn, sigmoid
from torch.optim import SGD
from torch.utils.data import Dataset


class LR(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(LR, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
        ).to(device)

    def init_weights_and_bias(self, weights, bias, device):
        weights = np.reshape(weights, (-1, weights.shape[0]))
        self.linear.weight.data = torch.Tensor(weights).to(device)
        self.linear.bias.data = torch.Tensor(np.asarray(bias)).to(device)

    def forward(self, x):
        # We will use BCEWithLogitsLoss. Do sigmoid outside function
        out = self.net(x)
        return out

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


class DependencyNetwork:
    def __init__(self, input_dim, output_dim, device, criterion, cfg):
        """
        Input is the vector of true_label and cnn predictions.
        Output is the vector of outputs of DN model
        """
        self.nns = [LR(input_dim - 1, 1, device) for _ in range(output_dim)]
        self.optimizers = [SGD(each_model.parameters(), lr=cfg.JOINT_LEARNING.LEARNING_RATE,
                               weight_decay=cfg.JOINT_LEARNING.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM,) for each_model in
                           self.nns]
        self.num_true_label = input_dim // 2
        self.criterion = criterion
        self.cfg = cfg

        # self.linear = nn.Linear(input_dim, output_dim)

    def to(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for each_model in self.nns:
            each_model.to(device)

    def set_lr(self, new_lr):
        for idx in range(len(self.optimizers)):
            for param_group in self.optimizers[idx].param_groups:
                param_group["lr"] = new_lr

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def train(self):
        for each_model in self.nns:
            each_model.train()

    def eval(self):
        for each_model in self.nns:
            each_model.eval()

    def forward_train(self, cnn_predictions, true_label, train, action="average", joint=False):
        """

        @param cnn_predictions:
        @param true_label:
        @param train:
        @param action:
        @param joint:
        @return:
        """
        # Code to get outputs while training
        batch_size = true_label.shape[0]
        outputs = torch.zeros((batch_size, len(self.nns)))
        # method - 2
        if joint:
            gradients_for_CNN = []
        # method - 3
        # gradients_for_CNN = torch.zeros((batch_size, len(self.nns))).to(device)
        total_loss = 0
        for each_true_label_index, this_nn in enumerate(self.nns):
            this_true_label = true_label[:, each_true_label_index]
            train_cnn_predictions = cnn_predictions.detach().clone()
            if joint:
                train_cnn_predictions.requires_grad = True
            if each_true_label_index == 0:
                input_to_lr = torch.cat((true_label[:, 1:], train_cnn_predictions), 1)
            elif each_true_label_index == self.num_true_label - 1:
                input_to_lr = torch.cat((true_label[:, :-1], train_cnn_predictions), 1)
            else:
                input_to_lr = torch.cat((true_label[:, :each_true_label_index],
                                         true_label[:, each_true_label_index + 1:], train_cnn_predictions), 1)

            # Use BCEwithLogits instead
            if train:
                self.optimizers[each_true_label_index].zero_grad()
            this_output = this_nn(input_to_lr)
            loss = self.criterion(torch.squeeze(this_output), this_true_label)
            outputs[:, each_true_label_index] = this_output.reshape((this_output.shape[0],))
            total_loss += loss.item()
            if train:
                torch.nn.utils.clip_grad_value_(
                        this_nn.parameters(), self.cfg.SOLVER.CLIP_GRAD_L2NORM
                )
                loss.backward()
                if joint:
                    gradient_wrt_train_cnn_predictions = train_cnn_predictions.grad
                # For every DN
                # method - 2
                    gradients_for_CNN.append(gradient_wrt_train_cnn_predictions)
                # method - 3
                # gradients_for_CNN += gradient_wrt_train_cnn_predictions
                self.optimizers[each_true_label_index].step()
        if train:
            # method 2
            if joint:
                tensor = torch.stack(gradients_for_CNN)
                if action.strip().lower() == "average":
                    # return gradients_for_CNN
                    return torch.mean(tensor, dim=0), total_loss
                    # method 3
                    # gradients_for_CNN = torch.divide(gradients_for_CNN, len(self.nns)).to(device)
                    # return gradients_for_CNN, total_loss
                    # return torch.mean(gradients_for_CNN, dim=0)
                elif action.strip().lower() == "sum":
                    # return gradients_for_CNN, total_loss
                    # tensor = torch.stack(gradients_for_CNN)
                    return torch.sum(tensor, dim=0), total_loss
                else:
                    raise Exception("Unkown action")
            else:
                return outputs, total_loss
        return outputs, total_loss

    def forward_sampling(self, x, cnn_predictions, var_sequence, device):
        """

        @param x: sample
        @param cnn_predictions: cnn_output
        @param var_sequence:
        @return:

        Args:
            device:
        """
        output_probs = torch.zeros_like(x)
        for each_true_label_index in var_sequence:
            if each_true_label_index == 0:
                input_to_lr = torch.cat((x[:, 1:], cnn_predictions), 1)
            elif each_true_label_index == self.num_true_label - 1:
                input_to_lr = torch.cat((x[:, :-1], cnn_predictions), 1)
            else:
                # print(x.shape, cnn_predictions.shape)
                input_to_lr = torch.cat((x[:, :each_true_label_index],
                                         x[:, each_true_label_index + 1:], cnn_predictions), 1)
            # input_to_lr.to(device)
            this_output = sigmoid(self.nns[each_true_label_index](input_to_lr))
            this_sample_prob = this_output.detach().cpu().numpy()
            # Check this sample prob shape - one dimension or two dimension
            this_sample = sample_from_probs_for_one_example(this_sample_prob)
            this_sample = torch.FloatTensor(this_sample).to(device)
            x[:, each_true_label_index] = this_sample.reshape((this_sample.shape[0],))
            output_probs[:, each_true_label_index] = this_output.reshape((this_output.shape[0],))

        return x, output_probs


@njit
def sample_from_probs_for_one_example(probabilities):
    output = np.zeros_like(probabilities)
    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            this_sample_prob = probabilities[i, j]
            random_num = np.random.random()
            if random_num <= this_sample_prob:
                this_sample_value = 1
            else:
                this_sample_value = 0
            # this_val = rand_choice_nb(np.array([0, 1]), prob=np.array([1 - this_sample_prob_1, this_sample_prob_1]))
            output[i, j] = this_sample_value
    return output


class AR_Dataset(Dataset):

    def __init__(self, X, y):
        """Initializes instance of class Dataset.

        """

        # Save target and predictors
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X[idx], self.y[idx]]
