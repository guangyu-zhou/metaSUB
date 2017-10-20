from __future__ import print_function

import numpy as np
import tensorflow as tf
from gen_data import *
import feature_extraction as fe

# Reserve
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_h = tf.Variable(tf.random_normal([n_in, n_out],mean = 0.0,stddev = 0.05))
        b_h = tf.Variable(tf.zeros([n_out]))

        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]

    def output(self):
        # print(self.input.shape, self.b)
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)

        return self.output

# output Layer
class OutputLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_o = tf.Variable(tf.random_normal([n_in, n_out], mean = 0.0, stddev = 0.05))
        b_o = tf.Variable(tf.zeros([n_out]))

        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        #changed relu to sigmoid
        self.output = tf.nn.sigmoid(linarg)

        return self.output

# model
def model(X, y_):
    X = X.astype(np.float32)
    h_layer = HiddenLayer(input = X, n_in = 25, n_out = 32)
    o_layer = OutputLayer(input = h_layer.output(), n_in = 32, n_out = 621)

    # loss function
    out = o_layer.output()
    # modified cross entropy to explicit mathematical formula of sigmoid cross entropy loss
    cross_entropy = -tf.reduce_sum( (  (y_*tf.log(out + 1e-9)) + ((1-y_) * tf.log(1 - out + 1e-9)) )  , name='xentropy' )

    # regularization
    l2 = (tf.nn.l2_loss(h_layer.w) + tf.nn.l2_loss(o_layer.w))
    lambda_2 = 0.01

    # compute loss
    loss = cross_entropy + lambda_2 * l2

    # compute accuracy for single label classification task
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float32"))

    return loss, accuracy

# Parameters

k_folds_Y, k_folds_loc = K_fold_split(3, 20, True)
test_Y = k_folds_Y[1]
test_locs = k_folds_loc[1]
train_Y = k_folds_Y[0]
train_locs = k_folds_loc[1]

line_vecs_train = fe.extract_feats(train_locs, False)
line_vecs_test = fe.extract_feats(test_locs, False)

print(line_vecs_train.shape, train_Y)

    # Run the initializer
    # sess.run()
loss, acc = model(line_vecs_train, train_Y)
print(loss, acc)



'''
learning_rate = 0.1
num_steps = 1000
batch_size = 30
display_step = 100

# Network Parameters
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
num_input = 191 # MNIST data input (img shape: 28*28)
num_classes = 621 # MNIST total classes (0-9 digits)

# Define the input function for training
k_folds_Y, k_folds_loc = K_fold_split(3, 20, True)
test_Y = k_folds_Y[1]
test_locs = k_folds_loc[1]
train_Y = k_folds_Y[0]
train_locs = k_folds_loc[1]

line_vecs_train = fe.extract_feats(train_locs, False)
line_vecs_test = fe.extract_feats(test_locs, False)

print(line_vecs_train.shape, train_Y)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'features': line_vecs_train}, y= np.array(train_Y),
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['features']
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.float64)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    #
    # # Evaluate the accuracy of the model
    print(pred_classes)
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    #
    # # TF Estimators requires to return a EstimatorSpec, that specify
    # # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
# model = tf.estimator.Estimator(model_fn)
# print("============Training============")
# model.train(input_fn, steps=num_steps)
'''




























































