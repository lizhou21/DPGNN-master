
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mymodel.layers import GraphConvolution, MLP, ADJRest, Attention,GraphConvforGCNII,MixedDropout,MixedLinear,MixHopConv
from mymodel.layers import ADJRest
import numpy as np
import scipy.sparse as sp




class DPGNN(nn.Module):#无log_softmax
    def __init__(self, args, nfeat, nhid, out, node_num):
        super(DPGNN, self).__init__()
        self.args = args
        self.create_LAM = ADJRest(args.topo_order + 1, 1, 1)
        self.create_fadj = ADJRest(self.args.fea_order + 1, 1, 1)
        self.fea_att = ADJRest(2, 1, 1)

        self.trans = nn.Linear(nfeat, nhid, bias=True)
        self.slayer = nn.Linear(nhid, out, bias=True)
        self.flayer = nn.Linear(nhid, out, bias=True)

        self.input_dropout = nn.Dropout(args.input_droprate)
        self.hidden_dropout = nn.Dropout(args.hidden_droprate)
        self.adj_dropout = nn.Dropout(args.adj_droprate)


    def fea_propa(self, s_adj, f_adj, X):
        X = self.input_dropout(X)
        X = F.relu(self.trans(X))
        X = self.hidden_dropout(X)
        X_s = self.slayer(X)
        X_f = self.flayer(X)
        s_adj = self.adj_dropout(s_adj)
        f_adj = self.adj_dropout(f_adj)
        X_s = torch.matmul(s_adj, X_s)  # (node_num,nclass)
        X_f = torch.matmul(f_adj, X_f)
        X_mix = torch.stack([X_s, X_f]).unsqueeze(0)
        if self.args.aggregator == 'att':
            X_mix = self.fea_att(X_mix).squeeze()
        elif self.args.aggregator == 'mean':
            X_mix = torch.mean(X_mix.squeeze(),dim=0)
        elif self.args.aggregator == 'max':
            X_mix = torch.max(X_mix.squeeze(), dim=0)[0]
        return X_mix



    def forward(self, X, A, training=True):#(n,n,9)
        # ① mix adj generation
        s_adj = A[0].unsqueeze(0).permute(0, 3, 1, 2)  # (1, edge_num, node_num, node_num)
        f_adj = A[1].unsqueeze(0).permute(0, 3, 1, 2)  # (1, edge_num, node_num, node_num)
        s_adj = self.create_LAM(s_adj).squeeze()  # （edge_num,node_num,node_num）
        f_adj = self.create_fadj(f_adj).squeeze()#(n,n)


        if training:
            X_out_list = []
            for i in range(self.args.sample):
                fea_out = self.fea_propa(s_adj, f_adj, X)
                X_out_list.append(fea_out)

            return X_out_list

        else:
            fea_out = self.fea_propa(s_adj, f_adj, X)
            return fea_out

