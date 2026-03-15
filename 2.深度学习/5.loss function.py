import torch
import torch.nn.functional as F

def binary_cross_entropy_loss(y_pred, y_true):
    probs = torch.sigmod(y_pred)

    epsilon = 1e-7
    probs = torch.clamp(probs, epsilon, 1 - epsilon)

    loss = - (y_true * torch.log(probs)) + (1 - y_true) * torch.log(1 - probs)

    return loss.mean()

