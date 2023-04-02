import argparse

def Config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
    parser.add_argument('--save_subpath', type=str, default='001', help='experiment id')
    parser.add_argument('--dataset', type=str, default='acm', help='Data set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=200, help='Patience')

    # my model
    parser.add_argument('--model_name', type=str, default='DPGNN', help='model choose')
    parser.add_argument('--adj_method', type=str, default='mix_adjs', help='adj generation.')
    parser.add_argument('--feature_normal', action='store_true', default=False, help='feature_normal_yes_or_no')
    parser.add_argument('--each_lable_num', type=str, default='20', help='model choose')

    parser.add_argument('--topk', type=int, default=12, help='top K for constructing feature graph')
    parser.add_argument('--topo_order', type=int, default=2, help='hop L for multi-hop topology graph')
    parser.add_argument('--fea_order', type=int, default=2, help='hop L for multi-hop feature graph')
    parser.add_argument('--sample', type=int, default=2, help='outputs S for self-ensembling')
    parser.add_argument('--tem', type=float, default=0.3, help='temperature T for sharpening function')
    parser.add_argument('--lam', type=float, default=0.7, help='coefficient lam for loss combination')
    parser.add_argument('--hidden', type=int, default=32, help='hidden size dh')
    parser.add_argument('--input_droprate', type=float, default=0.0, help='droprate for input layer.')
    parser.add_argument('--hidden_droprate', type=float, default=0.2, help='droprate for hidden layer.')
    parser.add_argument('--adj_droprate', type=float, default=0.5, help='droprate for hidden layer.')
    parser.add_argument('--aggregator', type=str, default='att', help='dual aggregator.')

    parser.add_argument('--droprate', type=float, default=0.5, help='general dropout')
    parser.add_argument('--loss', type=str, default='ce', help='loss function choose')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer for training')
    args = parser.parse_args()
    return args