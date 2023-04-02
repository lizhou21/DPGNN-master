import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodel.utils import *
import math



class ADJRest(nn.Module):
    def __init__(self, adj_nums, hop_channel, node_num):
        super(ADJRest, self).__init__()
        self.adj_nums = adj_nums
        self.hop_channel = hop_channel
        self.weight = nn.Parameter(torch.Tensor(hop_channel, adj_nums, node_num, 1))
        self.bias = None
        self.reset_parameters()

    def  reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A

