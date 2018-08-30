#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""learn who will win the game based on current board"""

import numpy as np
import tensorflow as tf
import _pickle as pickle


def a_model_fn(features, labels, mode, params):
    p1 = tf.reshape(features['player1'], [-1, 9])
    p2 = tf.reshape(features['player2'], [-1, 9])
    players = tf.concat([p1, p2], axis=1)
    hidden = tf.layers.dense(players, 24)
    predictions = tf.layers.dense(hidden, 2)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=predictions)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(1e-4, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, predictions),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def import_data(filename):
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out


def main():
    # training data (future import once we have python3 numpy arrays)
    try:
        array_player_1 = import_data('tensorflow_generate_data/player1.pkl')
        array_player_2 = import_data('tensorflow_generate_data/player2.pkl')
        outcome = import_data('tensorflow_generate_data/winLose.pkl')
    except UnicodeDecodeError:
        # example data form / placeholder
        # player 1, player 2
        # board for each player
        # 1 indicates where player has played
        # 0 indicates where player has not played
        # (so same field should not be 1 on both boards)
        array_player_1 = np.full((1000, 3, 3), 0, dtype=np.float32)
        array_player_2 = np.full((1000, 3, 3), 0, dtype=np.float32)
        array_player_1[0] = [[1, 0, 0], [1, 0, 1], [1, 1, 0]]
        array_player_2[0] = [[0, 1, 1], [0, 1, 0], [0, 0, 1]]

        # multi class label
        # [1, 0] player 1 wins
        # [0, 1] player 2 wins
        # [0, 0] draw
        outcome = np.full((1000, 2), 0, dtype=np.float32)
        outcome[0] = [1, 0]

    # tf estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"player1": array_player_1, "player2": array_player_2},
        y=outcome,
        num_epochs=1,
        batch_size=128,
        shuffle=False)

    nn = tf.estimator.Estimator(model_fn=a_model_fn, params={}, model_dir="/tmp/tictacfow_practice")

    for _ in range(20):
        nn.train(input_fn=train_input_fn, steps=160)


if __name__ == "__main__":
    main()
