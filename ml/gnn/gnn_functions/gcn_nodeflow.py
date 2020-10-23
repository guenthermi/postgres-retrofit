import dgl.function as fn
import mxnet as mx
import mxnet.gluon as gluon


class NodeUpdate(gluon.Block):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdate, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation
        self.old_features = None

    def forward(self, node):
        h = mx.nd.concat(self.old_features, node.data['h'], dim=1)
        h = self.dense(h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}

    def set_old_features(self, old_features):
        self.old_features = old_features


class Net(gluon.Block):
    def __init__(self, in_feats, num_hidden, num_cat, activation, dropout,
                 msg_fn, normalization, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.msg_fn = msg_fn
        self.normalization = normalization
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            self.layers.add(NodeUpdate(
                in_feats * 2, num_hidden[0], activation))
            for i in range(1, len(num_hidden)):
                self.layers.add(NodeUpdate(
                    num_hidden[i - 1] + in_feats, num_hidden[i], activation))
            self.layers.add(NodeUpdate(
                num_hidden[-1] + in_feats, num_cat, None))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)
            nf.layers[i].data['h'] = h
            self.layers[i].set_old_features(nf.layers[i + 1].data['features'])
            if (self.msg_fn == "mean"):
                nf.block_compute(i, fn.copy_src(src='h', out='m'), lambda node: {
                                 'h': node.mailbox['m'].sum(axis=1)}, layer)
            else:
                nf.block_compute(i, fn.copy_src(src='h', out='m'), lambda node: {
                                 'h': node.mailbox['m'].sum(axis=1)}, layer)
        h = nf.layers[-1].data.pop('activation')
        return h


def evaluate(pred, labels):
    indices = mx.nd.argmax(pred, axis=1)
    correct = mx.nd.sum(indices == labels)
    return (correct * 1.0 / len(labels)).asscalar()
