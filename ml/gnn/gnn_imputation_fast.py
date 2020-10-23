from collections import defaultdict
import numpy as np
import re
import random

import sys
import os
import time
import collections
import mxnet as mx
import networkx as nx
from copy import deepcopy
from mxnet import gluon

sys.path.insert(0, 'core/')
sys.path.insert(0, 'ml/')

import db_connection as db_con
import utils

from gnn_functions import prep_funcs as pr
from gnn_functions import gcn_nodeflow as gcn

import dgl
dgl.load_backend('mxnet')

GRAPH_FILE_PREFIX = '/tmp/graph_data'

VEC_TABLE_TEMPL = '{vec_table}'
VEC_TABLE1_TEMPL = '{vec_table1}'
VEC_TABLE2_TEMPL = '{vec_table2}'


def perform_train(g, batch_size, num_neighbors, num_hops, train_mask, net,
                  labels, loss_fcn, optimizer):
    """Executes the training in batches of samples from the provided training
    data."""
    # shuffle id set
    train_mask = train_mask.asnumpy()
    labels = labels.asnumpy()
    id_list = list(range(len(train_mask)))
    random.shuffle(id_list)
    train_data = []
    label_data = []
    train_data_set = set()
    for i in id_list:
        if train_mask[i] not in train_data_set:
            train_data.append(train_mask[i])
            train_data_set.add(train_mask[i])
            label_data.append(labels[i])
    index = 0
    train_data = mx.nd.array(train_data).astype(np.int64)
    label_data = mx.nd.array(label_data)
    for nf in dgl.contrib.sampling.NeighborSampler(g, batch_size, num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=False,
                                                   num_hops=num_hops,
                                                   seed_nodes=train_data):
        nf.copy_from_parent()
        with mx.autograd.record():
            pred = net(nf)
            batch_nids = nf.layer_parent_nid(-1).astype('int64')
            train_loss = loss_fcn(pred, label_data[index:min((index+batch_size), len(train_data))])
            train_loss = train_loss.sum() / len(batch_nids)
            index += batch_size
        train_loss.backward()
        optimizer.step(batch_size=1)
    return train_loss


def perform_test(g, num_neighbors, num_hops, test_mask, net, labels, loss_fcn):
    """Applies the classifier on the test set and returns prediction, loss,
    and accuracy value."""
    test_mask = test_mask.asnumpy()
    labels = labels.asnumpy()
    id_list = list(range(len(test_mask)))
    test_data = []
    label_data = []
    test_data_set = set()
    for i in id_list:
        if test_mask[i] not in test_data_set:
            test_data.append(test_mask[i])
            test_data_set.add(test_mask[i])
            label_data.append(labels[i])
    index = 0
    test_data = mx.nd.array(test_data).astype(np.int64)
    label_data = mx.nd.array(label_data)
    for nf in dgl.contrib.sampling.NeighborSampler(g, len(test_data),
                                                   num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=False,
                                                   num_hops=num_hops,
                                                   seed_nodes=test_data):
        nf.copy_from_parent()
        test_pred = net(nf)
        test_nids = nf.layer_parent_nid(-1).astype('int64')
        test_loss = loss_fcn(test_pred, label_data)
        test_loss = test_loss.sum() / len(test_nids)
    return test_pred, test_loss, label_data


def perform_validation(g, num_neighbors, num_hops, val_mask, net, labels):
    """Applies the classifier on the validatation set and returns the
    prediction and the accuracy value."""
    val_preds = []
    index = 0
    for nf in dgl.contrib.sampling.NeighborSampler(g, 1,
                                                   num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=False,
                                                   num_hops=num_hops,
                                                   seed_nodes=val_mask):
        nf.copy_from_parent()
        val_pred = net(nf)
        val_preds.append(gcn.evaluate(val_pred, mx.nd.array([labels[index].asnumpy()])))
        index += 1
    val_acc = np.mean(val_preds)
    return val_acc


def classify_fast_method(nodes, features, data_dict, text_lookup, g, train_size,
                         test_size, validation_size, epochs, iterations, lr,
                         activation, normalization, hidden_layers, dropout=0.2,
                         batch_size=256, num_neighbors=20, msg_fn='mean',
                         residual=False, stagnation=50):
    """
    Executes the training and evaluation of a classifiy for
    missing value imputation on a random sample several times.

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
        normalization (bool):
        hidden_layers (list): list of size values of hidden layers
        dropout (float): drop out rate
        bach_size (int): batch size for training
        msg_fn (function): aggregation function for messages in the
            message passing step  ('mean' or 'sum')
        stagnation (int): number of epochs with stagnating best loss value
            after that training is terminated
    Returns:
        list of accuracy values

    """

    num_labels = len(set(data_dict.values()))
    reverse_node_dict = collections.OrderedDict([(x[1], x[0]) for x in nodes])

    pr.prepare_features(g, features, normalization)
    g.readonly()
    val_accs, best_epochs, end_epochs, all_dur, all_test_losses, all_test_accs = [
    ], [], [], [], [], []
    for i in range(iterations):
        train_mask, train_labels, test_mask, test_labels, val_mask, val_labels = pr.create_masks(
            data_dict, text_lookup, reverse_node_dict, [train_size, test_size, validation_size])
        net = gcn.Net(features.shape[1], hidden_layers, num_labels, activation, dropout, msg_fn, normalization,
            prefix='GCN')
        net.initialize()
        loss_fcn = gluon.loss.SoftmaxCELoss()
        optimizer = gluon.Trainer(net.collect_params(), 'adam', {
                                  'learning_rate': lr, 'wd': 0})
        best_epoch, epoch, loss_stagnation, best_net, best_loss = 0, 0, 0, None, float('inf')
        dur, test_losses, test_accs = [], [], []
        while (epoch < epochs and loss_stagnation < stagnation):
            t0 = time.time()
            train_loss = perform_train(g, batch_size, num_neighbors, len(
                hidden_layers) + 1, train_mask, net, train_labels, loss_fcn, optimizer)
            test_pred, test_loss, label_data = perform_test(
                g, num_neighbors, len(hidden_layers) + 1, test_mask, net, test_labels, loss_fcn)
            mx.nd.waitall()
            dur.append(time.time() - t0)
            test_losses.append(test_loss.asscalar())
            test_acc = gcn.evaluate(test_pred, label_data)
            loss_stagnation = loss_stagnation + 1 if test_loss >= best_loss else 0
            test_accs.append(test_acc)
            epoch += 1

            if (test_loss < best_loss):
                best_loss = test_loss
                best_net = deepcopy(net)
                best_epoch = epoch
                print('new best loss', test_loss[0])
            print(("Iteration: {:01d} | Epoch {:03d} | Train Loss {:.4f} | " +
                   "Test Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}").format(
                i, epoch, train_loss.asscalar(), test_losses[-1], test_acc, dur[-1]))
        val_acc = perform_validation(g, num_neighbors, len(
            hidden_layers) + 1, val_mask, best_net, val_labels)
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

    # create edgelist
    graph_name = GRAPH_FILE_PREFIX + str(time.time())
    utils.create_edgelist(retro_config_filename,
                        db_config_file, graph_name)

    query = utils.get_vector_query(
        query, table_names, VEC_TABLE_TEMPL, VEC_TABLE1_TEMPL, VEC_TABLE2_TEMPL)

    id_lookup, text_lookup = pr.get_raw_data(
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

    g, nodes = pr.construct_graph(graph_name)
    features = pr.construct_features_from_vec_dict(
        nodes, text_lookup, vector_lookup)

    train_size = int(0.9 * train_data_size)
    test_size = train_data_size - train_size
    validation_size = test_data_size
    print(train_size, test_size, validation_size)

    scores = classify_fast_method(nodes=nodes, features=features,
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
    os.remove(graph_name + '.edges')
    os.remove(graph_name + '.terms')
    return mu, std, scores


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
