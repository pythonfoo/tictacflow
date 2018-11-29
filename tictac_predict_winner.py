#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""learn who will win the game based on current board"""

import numpy as np
import tensorflow as tf
import _pickle as pickle
import argparse




def a_model_fn(features, labels, mode, params):
    """very basic fully connected model"""
    p1 = tf.reshape(features['player1'], [-1, 9])
    p2 = tf.reshape(features['player2'], [-1, 9])
    players = tf.concat([p1, p2], axis=1)
    hidden = tf.layers.dense(players, 24, activation=tf.nn.relu)
    predictions = tf.layers.dense(hidden, 2)
    return score_n_spec(predictions, labels, mode)

def score_n_spec(predictions, labels, mode):
    """evaluation of predictions, including loss and making estimator spec"""
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=predictions)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(1e-5, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, tf.cast(predictions >= 0, tf.float32)),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def a_convolutional_model_fn(features, labels, mode, params):
    """very basic convolutional model"""
    p1 = tf.reshape(features['player1'], [-1, 3, 3, 1])
    p2 = tf.reshape(features['player2'], [-1, 3, 3, 1])
    players = tf.concat([p1, p2], axis=3)
    hidden1 = tf.layers.conv2d(players, filters=12, kernel_size=2, activation=tf.nn.relu)
    hidden2 = tf.layers.conv2d(hidden1, filters=16, kernel_size=2, activation=tf.nn.relu)
    predictions = tf.layers.conv2d(hidden2, filters=2, kernel_size=1)
    predictions = tf.reshape(predictions, [-1, 2])
    return score_n_spec(predictions, labels, mode)


def shared_weight_cnn_model_fn(features, labels, mode, params):
    p1 = tf.reshape(features['player1'], [-1, 3, 3, 1])
    p2 = tf.reshape(features['player2'], [-1, 3, 3, 1])
    with tf.variable_scope('player'):
        p1_hidden1 = tf.layers.conv2d(p1, filters=64, kernel_size=2, name='conv_1', activation=tf.nn.relu)
        p1_hidden1 = tf.layers.dropout(p1_hidden1, rate=0.9)
        p1_hidden2 = tf.layers.conv2d(p1_hidden1, filters=512, kernel_size=2, name='conv_2', activation=tf.nn.relu)
        p1_hidden2 = tf.layers.dropout(p1_hidden2, rate=0.9)
        p1_prediction = tf.layers.conv2d(p1_hidden2, filters=1, kernel_size=1, name='conv_3')

    with tf.variable_scope('player', reuse=True):
        p2_hidden1 = tf.layers.conv2d(p2, filters=64, kernel_size=2, name='conv_1', activation=tf.nn.relu)
        p2_hidden1 = tf.layers.dropout(p2_hidden1, rate=0.9)
        p2_hidden2 = tf.layers.conv2d(p2_hidden1, filters=512, kernel_size=2, name='conv_2', activation=tf.nn.relu)
        p2_hidden2 = tf.layers.dropout(p2_hidden2, rate=0.9)
        p2_prediction = tf.layers.conv2d(p2_hidden2, filters=1, kernel_size=1, name='conv_3')

    p1_prediction = tf.reshape(p1_prediction, [-1, 1])
    p2_prediction = tf.reshape(p2_prediction, [-1, 1])
    predictions = tf.concat([p1_prediction, p2_prediction], axis=1)
    print(p1_prediction.get_shape(), 'p1 h2')
    print(p2_prediction.get_shape(), 'p2 h2')
    print(predictions.get_shape())
    return score_n_spec(predictions, labels, mode)


def import_data(filename):
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out


def main(conv, train_dir):
    # training data
    try:
        array_player_1 = import_data('tensorflow_generate_data/random2_1000_games/player1.pkl')
        array_player_2 = import_data('tensorflow_generate_data/random2_1000_games/player2.pkl')
        outcome = import_data('tensorflow_generate_data/random2_1000_games/winLose.pkl')
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

    split_at = int(array_player_1.shape[0] * 0.8)

    # setup input functions
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"player1": array_player_1[:split_at], "player2": array_player_2[:split_at]},
        y=outcome[:split_at],
        num_epochs=None,
        batch_size=128,
        shuffle=True)

    dev_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"player1": array_player_1[split_at:], "player2": array_player_2[split_at:]},
        y=outcome[split_at:],
        num_epochs=1,
        batch_size=128,
        shuffle=True
    )

    # choose model fn
    if conv:
        model_fn = shared_weight_cnn_model_fn
    else:
        model_fn = a_model_fn

    # setup estimator (control class)
    nn = tf.estimator.Estimator(model_fn=model_fn, params={}, model_dir=train_dir)

    # actually train (and evaluate)
    for _ in range(20):  # to see the early part of the learning curve
        nn.train(input_fn=train_input_fn, steps=500)
        nn.evaluate(input_fn=train_input_fn, steps=50, name='training')
        nn.evaluate(input_fn=dev_input_fn, steps=50, name='dev')

    for _ in range(500):
        print('.')
        nn.train(input_fn=train_input_fn, steps=10000)
        nn.evaluate(input_fn=train_input_fn, steps=50, name='training')
        nn.evaluate(input_fn=dev_input_fn, steps=50, name='dev')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', action="store_true", help="use Convolutional instead of fully connected model")
    parser.add_argument('--train_dir', default='/tmp/tictacflow_practice',
                        help='directory to save model and training data')
    args = parser.parse_args()
    main(args.conv, args.train_dir)
