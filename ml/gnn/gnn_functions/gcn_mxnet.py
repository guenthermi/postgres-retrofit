import dgl.function as fn
import mxnet as mx
import mxnet.gluon as gluon
import numpy as np

gcn_msg = fn.copy_src(src='h', out='m')
gcn_mean_reduce = fn.mean(msg='m', out='neighbors')
gcn_sum_reduce = fn.sum(msg='m', out='neighbors')


class GCNLayer(gluon.Block):
    def __init__(self, in_feats, out_feats, activation, residual):
        super(GCNLayer, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation
        self.residual = None

    def forward(self, node):
        h = self.dense(node.data['activation'])
        if self.residual:
            h = mx.nd.add(node.data['features'], h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}

    def set_old_features(self, old_features):
        self.old_features = old_features


class GCN(gluon.Block):
    def __init__(self, in_feats, out_feats, activation, msg_fn, normalization,
                 residual, feature_dimension):
        super(GCN, self).__init__()
        self.msg_fn = msg_fn
        self.apply_mod = GCNLayer(
            in_feats + feature_dimension, out_feats, activation, residual)

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(gcn_msg, gcn_mean_reduce)
        g.ndata['activation'] = mx.nd.concat(
            g.ndata['features'], g.ndata['neighbors'], dim=1)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('activation')


class Net(gluon.Block):
    def __init__(self, in_feats, num_hidden, num_cat, activation, msg_fn,
                 normalization, residual, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            self.layers.add(GCN(
                in_feats, num_hidden[0], activation, msg_fn, normalization,
                residual, in_feats))
            for i in range(1, len(num_hidden)):
                self.layers.add(GCN(
                    num_hidden[i - 1], num_hidden[i], activation, msg_fn,
                    normalization, residual, in_feats))
            self.layers.add(
                GCN(num_hidden[-1], num_cat, None, msg_fn, normalization,
                    residual, in_feats))

    def forward(self, g):
        x = self.layers[0](g, g.ndata['features'])
        for i in np.arange(1, len(self.layers)):
            x = self.layers[i](g, x)
        return x


def evaluate(pred, labels):
    indices = mx.nd.argmax(pred, axis=1)
    correct = mx.nd.sum(indices == labels)
    return (correct * 1.0 / len(labels)).asscalar()
