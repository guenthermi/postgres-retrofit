import sys
import random
import re
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import time

from mxnet import nd
import networkx as nx


def get_raw_data(query, con, cur):
    """Returns two lookups: id_lookup and text_lookup. The id_lookup maps
    entity ids to labels and the text_lookup maps ids to text values."""
    cur.execute(query)
    id_lookup, text_lookup = construct_data_list(
        cur.fetchall())
    return id_lookup, text_lookup


def construct_data_list(query_result):
    id_lookup = dict()
    value_lookup = dict()
    text_lookup = dict()
    value_count = 0
    for elem in query_result:  # id value vec [vec2]
        if type(elem[2]) == type(None):
            continue
        if (len(elem) == 4) or (len(elem) == 5):
            value = elem[2]
            if value not in value_lookup:
                value_lookup[value] = value_count
                value_count += 1
            if elem[0] in id_lookup:
                print('Some data point is ambiguous:', elem)
                id_lookup[elem[0]].append(value_lookup[value])
            else:
                id_lookup[elem[0]] = [value_lookup[value]]
                text_lookup[elem[0]] = elem[1]
    id_lookup_clean = dict()
    for key in id_lookup:
        if len(set(id_lookup[key])) == 1:
            id_lookup_clean[key] = id_lookup[key][0]
    id_lookup = id_lookup_clean
    return id_lookup, text_lookup


def parse_nodes(name):
    f = open(name, "r", encoding="utf-8")
    content = f.read()
    f.close()
    nodes = re.split('\n', content)
    node_list = []

    for n in nodes:
        node = n.split(' ')
        if (node[0] != ''):
            node = (int(node[0]), node[1])
            node_list.append(node)

    return node_list


def construct_graph(name):
    """Creates a dgl graph from a file containing an edgelist and a file that
    maps ids used in the edgelist to node labels. The function returns
    the DGLGraph as well as a list of (id, node label) pairs."""
    g = dgl.DGLGraph()
    nx_graph = nx.read_edgelist(name + ".edges")
    edges = list(nx_graph.edges)
    nodes = parse_nodes(name + ".terms")
    src, dst = tuple(zip(*edges))
    g.add_nodes(len(nodes))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g, nodes


def get_random_data_samples(id_lookup, data_split):
    """Returns sampled dictonaries that map entity ids to labels for
    training, testing, and evaluation"""
    data_list = list(id_lookup)
    random.shuffle(data_list)
    limit1 = data_split[0]
    limit2 = data_split[0] + data_split[1]
    limit3 = sum(data_split)
    train_data = {key: val for key, val in id_lookup.items() if
                  key in data_list[:limit1]}
    test_data = {key: val for key, val in id_lookup.items() if
                 key in data_list[limit1:limit2]}
    val_data = {key: val for key, val in id_lookup.items() if
                key in data_list[limit2:limit3]}
    return train_data, test_data, val_data


def construct_features_from_vec_dict(nodes, text_lookup, vector_lookup):
    dim = len([x for x in list(vector_lookup.values()) if len(x) != 0][0][0])
    vectors = [vector_lookup[x[1]][0] if len(
        vector_lookup[x[1]]) > 0 else np.zeros(dim) for x in nodes]
    return torch.FloatTensor(vectors)


def prepare_features(g, features, normalization):
    """Adds features to the graph and normalize them if applicabe"""
    features = F.normalize(features, p=2, dim=1) if normalization else features
    features = nd.array(features)
    g.ndata['features'] = features


def prepare_train_test_split(id_lookup, text_lookup, reverse_node_dict, data_split):
    """Returns lists of (node_id, label_id) tuples."""
    train_data, test_data, val_data = get_random_data_samples(
        id_lookup, data_split)
    train_nodes = [(reverse_node_dict[text_lookup[id]], id_lookup[id])
                   for id in train_data]
    test_nodes = [(reverse_node_dict[text_lookup[id]], id_lookup[id])
                  for id in test_data]
    val_nodes = [(reverse_node_dict[text_lookup[id]], id_lookup[id])
                 for id in val_data]
    return train_nodes, test_nodes, val_nodes


def create_mx_arrays(train_nodes, test_nodes, val_nodes):
    train_ids = nd.array([x[0] for x in train_nodes]).astype(np.int64)
    train_labels = nd.array([x[1] for x in train_nodes])
    test_ids = nd.array([x[0] for x in test_nodes]).astype(np.int64)
    test_labels = nd.array([x[1] for x in test_nodes])
    val_ids = nd.array([x[0] for x in val_nodes]).astype(np.int64)
    val_labels = nd.array([x[1] for x in val_nodes])

    return train_ids, train_labels, test_ids, test_labels, val_ids, val_labels


def create_masks(id_lookup, text_lookup, reverse_node_dict, data_split):
    """Constructs training, testing, and validation datasets"""
    t0 = time.time()
    train_nodes, test_nodes, val_nodes = prepare_train_test_split(
        id_lookup, text_lookup, reverse_node_dict, data_split)
    train_ids, train_labels, test_ids, test_labels, val_ids, val_labels = create_mx_arrays(
        train_nodes, test_nodes, val_nodes)
    print("Prepared datasets in " + str(time.time() - t0) + "seconds")
    return train_ids, train_labels, test_ids, test_labels, val_ids, val_labels
