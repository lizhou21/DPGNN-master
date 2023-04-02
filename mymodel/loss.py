import torch
import torch.nn as nn
from torch.nn import functional as F


class NLL_loss():
    def __init__(self):
        self.crit = nn.NLLLoss()

    def __call__(self, logits, targets, index, epoch):
        logits = torch.log_softmax(logits, -1)
        loss = self.crit(logits[index], targets[index])
        return loss


class CrossEntropy():
    def __init__(self):
        self.crit = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, index, epoch):
        loss = self.crit(logits[index], targets[index])
        return loss
