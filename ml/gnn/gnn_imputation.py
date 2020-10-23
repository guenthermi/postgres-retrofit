from collections import defaultdict
import numpy as np
import re

import sys
import time
import collections
import mxnet as mx
import networkx as nx
from copy import deepcopy
from mxnet import gluon

sys.path.insert(0, 'core/')
sys.path.insert(0, 'ml/')

from gnn_functions import prep_funcs as pr
from gnn_functions import gcn_mxnet as gcn

import db_connection as db_con
import utils


GRAPH_FILE_PREFIX = '/tmp/graph_data'

VEC_TABLE_TEMPL = '{vec_table}'
VEC_TABLE1_TEMPL = '{vec_table1}'
VEC_TABLE2_TEMPL = '{vec_table2}'


def perform_train(g, train_mask, labels, net, loss_fcn, optimizer):
    """Executes a training set on the provided training data."""
    with mx.autograd.record():
        pred = net(g)
        train_loss = loss_fcn(pred[train_mask], labels)
        train_loss = train_loss.sum() / len(train_mask)
    train_loss.backward()
    optimizer.step(batch_size=1)
    return pred, train_loss


def perform_validation(g, val_mask, labels, net):
    """Applies the classifier on the validatation set and returns
    the predition and the accuracy value."""
    val_pred = net(g)
    val_acc = gcn.evaluate(val_pred[val_mask], labels)
    return val_pred, val_acc


def classify_exact_method(nodes, features, data_dict, text_lookup, g,
                          train_size, test_size, validation_size, epochs,
                          iterations, lr, activation, normalization,
                          hidden_layers, msg_fn='mean', residual=False,
                          stagnation=20):
    """
    Executes the training and evaluation of a classifiy
    for missing value imputation on a random sample several times.

    Attributes:
        nodes: list of nodes in the graph consisting of tuples of
            nodeId and nodeName
        features: list of features coresponding to the nodes
        data_dict: dictonary assigning nodes to labels
        g (dgl.graph.DGLGraph): graph used for graph neural network
        train_size (int): number of samples used for training
        test_size (int): number of samples used for testing
        validation_size (int): number of samples used for validation
        epochs (int): maximal number of epochs
        iterations (int): number of executings
        lr (float): learning rate for the optimizer
        activation (function): mx.nd.relu or mx.nd.sigmoid
        normalization (bool): feature vectors are normalized before
            the classification if set to True
        hidden_layers (list): list of size values of hidden layers
        msg_fn (function): aggregation function for messages in the
            message passing step  ('mean' or 'sum')
        residual: use residual learing if set to True
        stagnation (int): number of epochs with stagnating best loss value
            after that training is terminated
    Returns:
        list of accuracy values

    """

    print('train_size', train_size)
    print('test_size', test_size)
    print('validatation_size', validation_size)
    num_labels = len(set(data_dict.values()))
    reverse_node_dict = collections.OrderedDict([(x[1], x[0]) for x in nodes])
    pr.prepare_features(g, features, normalization)
    val_accs, best_epochs, end_epochs, all_dur, all_test_losses, all_test_accs = [
    ], [], [], [], [], []
    for i in range(iterations):
        train_mask, train_labels, test_mask, test_labels, val_mask, val_labels = pr.create_masks(
            data_dict, text_lookup, reverse_node_dict, [train_size, test_size, validation_size])
        print('hidden_layers', features.shape[1], hidden_layers, num_labels, activation)
        net = gcn.Net(features.shape[1], hidden_layers, num_labels,
                      activation, msg_fn, normalization, residual, prefix='GCN')
        net.initialize()
        loss_fcn = gluon.loss.SoftmaxCELoss()
        optimizer = gluon.Trainer(net.collect_params(), 'adam', {
                                  'learning_rate': lr, 'wd': 0})
        best_epoch, epoch, loss_stagnation, best_net, best_loss = 0, 0, 0, None, float('inf')
        dur, test_losses, test_accs = [], [], []
        while (epoch < epochs and loss_stagnation < stagnation):
            t0 = time.time()
            pred, train_loss = perform_train(
                g, train_mask, train_labels, net, loss_fcn, optimizer)
            test_loss = loss_fcn(pred[test_mask], test_labels)
            test_loss = test_loss.sum() / len(test_mask)
            mx.nd.waitall()
            dur.append(time.time() - t0)
            test_losses.append(test_loss.asscalar())
            test_acc = gcn.evaluate(pred[test_mask], test_labels)
            loss_stagnation = loss_stagnation + 1 if test_loss >= best_loss else 0
            test_accs.append(test_acc)
            epoch += 1

            if (test_loss < best_loss):
                best_loss = test_loss
                best_net = deepcopy(net)
                best_epoch = epoch
            print(("Iteration: {:01d} | Epoch {:03d} | Train Loss {:.4f} | " +
                   "Test Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}").format(
                i, epoch, train_loss.asscalar(), test_losses[-1], test_acc, dur[-1]))
        val_pred, val_acc = perform_validation(
            g, val_mask, val_labels, best_net)
        val_accs.append(val_acc)
        best_epochs.append(best_epoch)
        end_epochs.append(epoch)
        all_dur.append(dur)
        all_test_losses.append(test_losses)
        all_test_accs.append(test_accs)
        del train_mask, test_mask, val_mask, net, best_net
        print("Best Epoch: {:03d} | End_Epoch: {:03d} | Val_Acc_Best: {:.4f}".format(
            best_epoch, epoch, val_accs[-1]))
    return val_accs


def main(argc, argv):

    if argc < 8:
        print('Not enough arguments')
        return

    retro_config_filename = argv[1]
    db_config_file = argv[2]
    query = argv[3]  # query that returns tuple of relations which are present
    train_data_size = int(argv[4])
    test_data_size = int(argv[5])
    iterations = int(argv[6])
    table_names = []
    table_names.append(argv[7])
    if argc == 9:
        table_names.append(argv[8])

    db_config = db_con.get_db_config(db_config_file)
    con, cur = db_con.create_connection(db_config)

    # create edgelist from group_file
    utils.create_edgelist(retro_config_filename,
                        db_config_file, GRAPH_FILE_PREFIX)

    query = utils.get_vector_query(
        query, table_names, VEC_TABLE_TEMPL, VEC_TABLE1_TEMPL, VEC_TABLE2_TEMPL)

    id_lookup, text_lookup  = pr.get_raw_data(
        query, con, cur)

    vector_lookup = defaultdict(list)

    print('table_names', table_names)

    vectors = list()
    for table_name in table_names:
        vectors.append(utils.get_db_vectors(table_name, con, cur))

    for key in vectors[0]:
        combined_vector = []
        complete = True
        for vector_dataset in vectors:
            if key not in vector_dataset:
                complete = False
                print('WARNING: Ignore', key, 'since it is not present in all vector datasets')
                break
            if np.linalg.norm(vector_dataset[key]) > 0:
                combined_vector.append(vector_dataset[key] / np.linalg.norm(vector_dataset[key]))
            else:
                combined_vector.append(vector_dataset[key])
        if complete:
            combined_vector = np.concatenate(combined_vector)
            vector_lookup[key].append(combined_vector / np.linalg.norm(combined_vector))

    g, nodes = pr.construct_graph(GRAPH_FILE_PREFIX)
    features = pr.construct_features_from_vec_dict(
        nodes, text_lookup, vector_lookup)

    train_size = int(0.9 * train_data_size)
    test_size = train_data_size - train_size
    validation_size = test_data_size

    scores = classify_exact_method(nodes=nodes, features=features,
                                   data_dict=id_lookup, text_lookup=text_lookup,
                                   g=g, train_size=train_size,
                                   test_size=test_size,
                                   validation_size=validation_size, epochs=500,
                                   iterations=iterations, lr=0.001,
                                   activation=mx.nd.relu, normalization=True,
                                   hidden_layers=[300], msg_fn='mean',
                                   stagnation=50)

    mu = np.mean(scores)
    std = np.std(scores)
    print('mu', mu, 'std', std, 'scores', scores)

    return mu, std, scores


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
