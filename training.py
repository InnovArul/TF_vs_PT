import os
import logging

import numpy as np
import tensorflow as tf
import torch
from torch.nn.functional import one_hot

from logistic_class import Logistic
from image_loader import load_imagefolder
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(1234)

# This class contains 4 methods: 1-train_loop, 2-val_loop, 3-sgd, 4-logistic_regression
# the only reasons why I created a class is to expertise OOP and to create consistency
# to avoid adding an extra-parameter "lib" for each of the functions.
# Note that while initializing this class, I initialize the Logistic one too since both need only the
# parameter "library" (obviously you could have created all the method within just one class which would
# have been more elegant)


class TrainModel:
    def __init__(self, library):
        self.library = library
        self.func = Logistic(library)

    def sgd(self, params, grads, lr, bs):
        if self.library == "tf":

            for param, grad in zip(params, grads):
                # each ordered item W & b is updated by the corresponding element of grads[0] and grads[1]
                param.assign_sub(lr * grad / bs)
        else:
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    param -= (lr * grad / bs)

    def train_loop(self, lr, train_data, W, b):
        if self.library == "tf":
            losses = []
            accuracies = []
            for X, Y in train_data:
                with tf.GradientTape() as tape:
                    # forward pass
                    X = X / 255.0
                    # y_hat has shape (batch_size, num_of_classes)
                    y_hat = self.logistic_regression(X, W, b)
                    # calculate loss
                    # one_hot_encoding
                    one_hot = tf.one_hot(Y, 43)
                    # let's say target = [y1, y2] and sources = [x1, x2]. The result will be:
                    # grad will be grad = [dy1/dx1 + dy2/dx1, dy1/dx2 + dy2/dx2]
                    loss = self.func.cross_entropy(y_hat, one_hot)
                    losses.append(tf.math.reduce_mean(loss))

                    # how tape.gradient works? It works thanks to the tensor graph that is created
                    # while running the program. Which means that the tensor graph tracks the
                    # creation of the variable "loss" and knows that its value depends on other
                    # variables, and therefore the gradient between loss and W or b can be computed
                    # grad--> is composed by two lists: grads[0] has shape (3072, 43) as W, grads[1]
                    # has shape (43,) as b
                    grads = tape.gradient(loss, [W, b])
                    self.sgd([W, b], grads, lr, X.shape[0])

                    acc = self.func.accuracy(y_hat, Y)
                    accuracies.append(acc)
        else:
            losses = []
            accuracies = []
            for X, Y in train_data:
                X = X / 255
                y_hat = self.logistic_regression(X, W, b)
                one_hot = torch.nn.functional.one_hot(Y, 43).bool()
                loss = self.func.cross_entropy(y_hat, one_hot)
                losses.append(torch.mean(loss).item())
                grads = torch.autograd.grad(loss, [W,b], grad_outputs=torch.ones_like(loss))
                self.sgd([W, b], grads, lr, X.shape[0])
                acc = self.func.accuracy(y_hat, Y)
                accuracies.append(acc)
        return np.mean(losses), np.mean(accuracies)

    def logistic_regression(self, X, W, b):
        if self.library == "tf":
            # flatten_X has shape (batch_size, W*H*Channels) --> in our case (64, 32*32*3)
            flatten_X = tf.reshape(X, (-1, W.shape[0]))
            out = self.func.softmax(tf.matmul(flatten_X, W) + b)
        else:
            flatten_X = X.reshape((-1, W.shape[0]))
            out = self.func.softmax(torch.matmul(flatten_X,W) + b)
        return out

    def val_loop(self, data, W, b):
        accuracy = []
        for X, Y in data:
            X = X / 255.0
            out = self.logistic_regression(X, W, b)
            acc = self.func.accuracy(out, Y)
            accuracy.append(acc)
        return np.mean(accuracy)


def logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()

    #FORMAT = logging.Formatter("%(asctime)s, %(message)s")
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    logger = logger(__name__)
    lib = "tf"
    train_set, val_set = load_imagefolder("../workspace_7/GTSRB/Final_Training/Images/", 0.1, lib)
    train_class = TrainModel(lib)
    epochs = 10
    lr = 0.1

    X, y = next(iter(train_set))

    # number of classes of the dataset
    num_outputs = 43
    if train_class.library == "tf":
        num_inputs = tf.reduce_prod(X.shape[1:], 0)
        W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                         mean=0, stddev=0.01))
        b = tf.Variable(tf.zeros(num_outputs))
    else:
        #t = torch.tensor(X.shape)
        #num_inputs = torch.prod(t[np.r_[0,2:]])
        num_inputs = torch.prod(torch.tensor(X.shape)[1:], 0)
        W = torch.normal(mean=0, std=0.01, size=(num_inputs, num_outputs), requires_grad=True)
        b = torch.zeros(num_outputs, requires_grad=True)
    for epoch in range(1, epochs+1):

        logger.info('Epoch {}'.format(epoch))

        loss, acc = train_class.train_loop(lr, train_set, W, b)

        logger.info('Mean training loss: {:1f}, mean training accuracy {:1f}'.format(loss, acc))

        val_acc = train_class.val_loop(val_set, W, b)

        logger.info('Mean validation accuracy {:1f}'.format(val_acc))

