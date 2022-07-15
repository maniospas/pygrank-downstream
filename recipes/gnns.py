import pygrank as pg
import tensorflow as tf


class APPNP(tf.keras.Sequential):
    def __init__(self, num_inputs, num_outputs, hidden=64, ranker=None):
        super().__init__([
            tf.keras.layers.Dropout(0.5, input_shape=(num_inputs,)),
            tf.keras.layers.Dense(hidden, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(1.E-5)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
        ])
        self.ranker = pg.PageRank(0.9, renormalize=True,
                                  assume_immutability=True,
                                  use_quotient=False,
                                  error_type="iters", max_iters=10) if ranker is None else ranker
        self.input_spec = None  # prevents some versions of tensorflow from checking call inputs

    def call(self, inputs, training=False):
        graph, features = inputs
        predict = super().call(features, training=training)
        predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
        return tf.nn.softmax(predict, axis=1)


def node_classification(graph, known_members_set, node_features):
    labels = pg.combine_cols([pg.to_signal(graph, known_members) for known_members in known_members_set])
    nodes = [v for v in range(len(graph)) if pg.sum(labels[v, :]) != 0]
    training, validation = pg.split(nodes, 0.8)
    with pg.Backend('tensorflow'):
        with tf.device('/GPU:1'):
            model = APPNP()
            pg.gnn_train(model, graph, node_features, labels, training, validation, verbose=True)
            predictions = model([graph, node_features])
    ranks_set = [pg.to_signal(graph, ranks) for ranks in pg.separate_cols(predictions)]
    found_set = [list() for _ in known_members_set]
    options = list(range(len(ranks_set)))
    for v in graph:
        found_set[max(options, key=lambda i:ranks_set[i][v])].append(v)