#! /usr/bin/env python

import tensorflow as tf
import pickle
import numpy as np
import os
import time
import datetime
import data_helpers
from model import Model
from tensorflow.contrib import learn
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, label_ranking_loss


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_integer("n_fold", 3, "number of folds (3/5 default: 3)")
tf.flags.DEFINE_integer("fold", 0, "(0 ~ n_fold-1 default: 0)")

# Model Hyperparameters
tf.flags.DEFINE_integer("num_hidden_layers", 1, "Number of hidden layers (default: 1)")
tf.flags.DEFINE_integer("num_hidden_neurons", 8, "Number of hidden layers (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-3, "L2 regularization lambda (default: 1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 1000)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def calc_precision_recall(yt, yp):
    pu, pd = 0.0, 0.0
    ru, rd = 0.0, 0.0
    assert(len(yt) == len(yp))
    for i in range(len(yt)):
        if int(yp[i]) == 1:
            pd += 1.0
        if int(yt[i]) == 1:
            rd += 1.0
        if int(yt[i]) == 1 and int(yp[i]) == 1:
            pu += 1.0
            ru += 1.0
    return pu/pd if pd > 0.0 else 0.0, ru/rd if rd > 0.0 else 0.0



def evaluate(_y_true, _y_pred, _y_scores):
    y_true = np.array(_y_true)
    y_pred = np.array(_y_pred)
    y_scores = np.array(_y_scores)
    
    pre = precision_score(y_true, y_pred,average='micro')
    rec = recall_score(y_true, y_pred,average='micro')
    fs = f1_score(y_true, y_pred,average='micro')
    hl = hamming_loss(y_true, y_pred)
    rl = label_ranking_loss(y_true, y_scores)
    return pre, rec, fs, hl, rl


if __name__ == '__main__':
    # Load parameters
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    n_fold = FLAGS.n_fold
    fold = FLAGS.fold
    num_hidden_layers = FLAGS.num_hidden_layers
    num_hidden_neurons = FLAGS.num_hidden_neurons

    x_train = np.load('../data/{}fold/train_X_{}.data'.format(n_fold, fold),encoding='latin1')
    y_train = np.load('../data/{}fold/train_Y_{}.data'.format(n_fold, fold),encoding='latin1')
    x_test = np.load('../data/{}fold/test_X_{}.data'.format(n_fold, fold),encoding='latin1')
    y_test = np.load('../data/{}fold/test_Y_{}.data'.format(n_fold, fold),encoding='latin1')

    assert(x_train.shape[0] == y_train.shape[0])
    assert(x_test.shape[0] == y_test.shape[0])
    assert(x_train.shape[1] == x_test.shape[1])
    assert(y_train.shape[1] == y_test.shape[1])

    num_features = x_train.shape[1]
    num_labels = y_train.shape[1]
    print(num_features, num_labels)   

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Model(
                    num_features=num_features,
                    num_labels=num_labels,
                    num_hidden_layers=num_hidden_layers,
                    num_hidden_neurons=num_hidden_neurons)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, predictions, scores = sess.run(
                    [train_op, global_step, model.loss, model.accuracy, model.predictions, model.scores],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #   print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                #    print(evaluate(y_batch, predictions, scores))
            
            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, predictions, scores = sess.run(
                    [global_step, model.loss, model.accuracy, model.predictions, model.scores],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                return evaluate(y_batch, predictions, scores)
#                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f = 0.0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                pre, rec, fs, hl, rl = dev_step(x_test, y_test)
                if fs > best_f:
                    best_f = fs
                    print(pre, rec, fs, hl, rl)

