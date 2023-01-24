import torch
from torchviz import make_dot
model = torch.load("./pre_trained/SlowFast.pyth")
image = make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")