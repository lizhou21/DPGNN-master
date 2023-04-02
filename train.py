from __future__ import division
from __future__ import print_function
import os
import time
import pickle as pkl
from mymodel.config import Config
from mymodel.helper import ensure_dir, save_config, FileLogger
from mymodel.adj_generate import *
from mymodel.utils import *


# Training settings
args = Config()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)




# load data
dataset = args.dataset


with open('./data/processed/{}.data'.format(dataset), 'rb') as file:
    data = pkl.load(file)
features, labels, graphs, idx_train, idx_val, idx_test = data['features'], data['labels'], data['graph'], data['train'+args.each_lable_num], data['val'+args.each_lable_num], data['idx_test']

node_num = features.shape[0]
infeat = features.shape[1]
nclass = labels.shape[1]



if args.feature_normal:
    features = normalize(features)

features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.argmax(labels, -1))
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

adj_method = adj_fetch(args.adj_method)

A = adj_method(graphs, features, args)



# define model save dir
model_save_dir = 'save_models/' + args.dataset + '/' + str(args.save_subpath)
ensure_dir(model_save_dir, verbose=True)
save_config(vars(args), model_save_dir + '/config.json', verbose=True)
model_name = dataset +'.pt'
save_path = os.path.join(model_save_dir, model_name)



def train(epoch):
    t = time.time()
    X = features # node feature
    adj = A # graph structure
    model.train()
    optimizer.zero_grad()
    output = model(X, adj, model.training)


    loss_train = 0
    for k in range(args.sample):
        loss_train += criterion(output[k], labels, idx_train, epoch)
    if 1:
        loss_train = loss_train / args.sample
        loss_consis = consis_loss([F.softmax(o, dim=-1) for o in output], args.tem, args.lam)
        loss_train = loss_train + loss_consis
    acc_train = accuracy(F.log_softmax(output[0], dim=-1)[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(X, adj, model.training)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(F.log_softmax(output, dim=-1)[idx_val], labels[idx_val])

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(F.log_softmax(output, dim=-1)[idx_test], labels[idx_test])
    f1_test = f1_compute(F.log_softmax(output, dim=-1)[idx_test], labels[idx_test])

    file_logger.log("{:04d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}s\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch + 1,
                                                                             loss_train.item(),
                                                                             acc_train.item(),
                                                                             loss_val.item(),
                                                                             acc_val.item(),
                                                                             time.time() - t, loss_test.item(), acc_test.item(), f1_test.item()))

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'loss_test:{:.4f}'.format(loss_test.item()),
          'acc_test:{:.4f}'.format(acc_test.item()),
          'f1_test:{:.4f}'.format(f1_test.item()))
    return loss_train.item(), loss_val.item(), acc_val.item(), loss_test.item(), acc_test.item()

def Train():
    # Train model
    t_total = time.time()
    loss_train_values = []
    loss_values = []
    acc_values = []
    loss_tests = []
    acc_tests = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        loss_train, l, a, loss_t, acc_t = train(epoch)
        loss_train_values.append(loss_train)
        loss_values.append(l)
        acc_values.append(a)

        loss_tests.append(loss_t)
        acc_tests.append(acc_t)

        file_logger.log(bad_counter)
        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            file_logger.log("Early stop! Min loss:{},  Max accuracy:{}".format(loss_mn, acc_mx))
            file_logger.log("Early stop model validation loss:{},  accuracy:{}".format(loss_best, acc_best))

            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    file_logger.log("Optimization Finished!")
    file_logger.log("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    file_logger.log('test max acc:{:.4f}'.format(max(acc_tests)))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('test max acc:{:.4f}'.format(max(acc_tests)))

    # Restore best model
    file_logger.log("Loading {}th epoch".format(best_epoch))
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(save_path))

def test():
    model.eval()
    X = features
    adj = A
    output = model(X, adj, model.training)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(F.log_softmax(output, dim=-1)[idx_test], labels[idx_test])
    f1_test = f1_compute(F.log_softmax(output, dim=-1)[idx_test], labels[idx_test])

    file_logger.log('stru_node_to_hop attention:{}'.format(F.softmax(model.create_LAM.weight, dim=1).squeeze().permute(1, 0)[0:100].data))
    file_logger.log('fea_node_to_hop attention:{}'.format(F.softmax(model.create_fadj.weight, dim=1).squeeze().permute(1, 0)[0:100].data))
    file_logger.log('perception attention:{}'.format(F.softmax(model.fea_att.weight, dim=1).squeeze().data))

    print('stru_node_to_hop attention:{}'.format(F.softmax(model.create_LAM.weight, dim=1).squeeze().permute(1, 0)[0:100].data))
    print('fea_node_to_hop attention:{}'.format(F.softmax(model.create_fadj.weight, dim=1).squeeze().permute(1, 0)[0:100].data))
    print('dual-perception attention:{}'.format(F.softmax(model.fea_att.weight, dim=1).squeeze().data))

    file_logger.log("Test set results: loss= {:.4f} accuracy= {:.4f} F1_score= {:.4f}".format(loss_test.item(), acc_test.item(), f1_test.item()))

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "F1_score= {:.4f}".format(f1_test.item()))





for seed in range(0, 100):
    print('seed:{}'.format(seed))
    print("===================================")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    file_logger = FileLogger(model_save_dir + '/seed_' + str(seed),
                             header='# epoch\tloss_train\tacc_train\tloss_val\tacc_val\ttime\tacc_test\tf1_test')

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))

    # define model
    model = get_model(args, infeat, nclass, node_num)
    criterion = get_loss(args, labels = labels, num_classes=nclass)
    optimizer = get_optimizer(model, args)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        if args.adj_method == 'mix_adjs':
            A = [adj.cuda() for adj in A]
        else:
            A = A.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    Train()
    test()
