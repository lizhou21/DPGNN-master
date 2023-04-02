 import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import os
from collections import *

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adj_normalize(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):#解析获取index
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str = 'cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = [] # 依次存放各文件中的数据
    for i in range(len(names)):
        with open("data/raw/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/raw/{}/ind.{}.test.index".format(dataset_str, dataset_str)) #获取test data index
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder ) +1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended


    obj = {}
    # ① obtain raw features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    obj['features'] = features

    # ② obain labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot
    obj['labels'] = labels

    # ③ obtain index
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()
    obj['idx_train'] = idx_train
    obj['idx_val'] = idx_val
    obj['idx_test'] = idx_test

    # ④ obtain graph adj
    obj['graph'] = {}
    obj['graph']['origin-hop'] = graph
    print('1-hop edges:{}'.format(edge_statistics(graph)))
    order_adj = [defaultdict(list) for i in range(10)]
    G = nx.Graph(graph)
    spl = list(nx.all_pairs_shortest_path_length(G))
    for nodei, nodej_list in enumerate(spl):
        for nodej, d in nodej_list[1].items():
            if 1 <= d <= 10:
                order_adj[d - 1][nodei].append(nodej)

    for i, hop_adj in enumerate(order_adj):
        name = str(i + 1) + '-hop'
        obj['graph'][name] = hop_adj
        print('{}-hop edges:{}'.format(i+1, edge_statistics(hop_adj)))


    return obj

def edge_statistics(adj):
    num = 0
    for e in adj.values():
        num = num + len(set(e))
    return num

# file_name = './data/raw'
# out_file = './data/processed/'

# file = os.path.join(out_file, 'citeseer.data')

dataset = 'cora'
data = load_data(dataset)
save_file = open('./data/processed/{}.data'.format(dataset), 'wb')
pkl.dump(data, save_file)
print('a')