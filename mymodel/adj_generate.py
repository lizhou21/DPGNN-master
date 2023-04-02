import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import scipy.sparse as sp
from mymodel.utils import *
import networkx as nx
from mymodel.model import *
import math
# from mymodel.utils import *
from mymodel.loss import *
from sklearn.metrics.pairwise import cosine_similarity as cos

def get_model(args, infeat, nclass, node_num):
    if args.model_name == 'DPGNN':
        model = DPGNN(args, infeat, args.hidden, nclass, node_num)
    return model


def get_loss(args, labels=None, num_classes=10):
    if args.loss == 'ce':
        criterion = CrossEntropy()
    elif args.loss == 'nl':
        criterion = NLL_loss()
    else:
        raise KeyError("Loss `{}` is not supported.".format(args.loss))

    return criterion


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        print("Using `SGD` optimizer")
        return optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.optimizer == 'adam':
        print("Using `Adam` optimizer")
        if args.model_name == 'GCNII':
            return optim.Adam([
                        {'params': model.params1, 'weight_decay': args.wd1},
                        {'params': model.params2, 'weight_decay': args.wd2},
                        ], lr = args.lr)

        return optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    else:
        raise KeyError("Optimizer `{}` is not supported.".format(args.optimizer))




def adj_fetch(type):
    switcher = {
        'mix_adjs': mix_adjs,
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func

def get_multi_hop_sparse(adj, order):
    adj_list = []
    adj_tmp = adj.to_dense()
    adj_list.append(torch.eye(adj_tmp.shape[0]).type(torch.FloatTensor))
    adj_list.append(adj_tmp)
    for i in range(1, order):
        adj_tmp = torch.mm(adj, adj_tmp)
        adj_list.append(adj_tmp)
    adj_list = torch.stack(adj_list, dim=-1)
    return adj_list

def get_multi_hop(adj, order):
    adj_list = []
    adj_tmp = adj
    adj_list.append(torch.eye(adj_tmp.shape[0]).type(torch.FloatTensor))
    adj_list.append(adj_tmp)
    for i in range(1, order):
        adj_tmp = torch.mm(adj, adj_tmp)
        adj_list.append(adj_tmp)
    adj_list = torch.stack(adj_list, dim=-1)
    return adj_list


def mix_adjs(graphs, feature, args):
    # origin_hop = graphs['origin-hop']
    origin_hop = graphs
    graph_edge = np.array(nx.from_dict_of_lists(origin_hop).edges)
    adj = sp.coo_matrix((np.ones(graph_edge.shape[0]), (graph_edge[:, 0], graph_edge[:, 1])), shape=(feature.shape[0], feature.shape[0]),
                             dtype=np.float32)

    node_num = adj.shape[0]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])  # 此操作，不会重复add对角线上已有的1
    adj = adj_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    stru_adj = get_multi_hop_sparse(adj, args.topo_order)

    fadj = torch.zeros((feature.shape[0], feature.shape[0]))
    s = cos_sim(feature, feature)
    index = torch.topk(s, k=args.topk, dim=1)[1]
    fadj.scatter_(1, index, 1)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)  # 对称
    D = torch.diag(fadj.sum(1) ** (-0.5))
    fadj = torch.mm(D, fadj)
    fadj = torch.mm(fadj, D)
    fea_adj = get_multi_hop(fadj, args.fea_order)
    return [stru_adj, fea_adj]



