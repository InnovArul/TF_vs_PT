import numpy as np
import argparse
import logging
import tensorflow as tf
from dataset import get_datasets
from logistic import softmax, cross_entropy, accuracy
import os
import sys
sys.path.append(os.path.abspath('.'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def sgd(params, grads, lr, bs):
    """
    stochastic gradient descent implementation
    args:
    - params [list[tensor]]: model params
    - grad [list[tensor]]: param gradient
    - lr [float]: learning rate
    - bs [int]: batch_size
    """
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / bs)


def training_loop(lr):
    """
    training loop
    args:
    - lr [float]: learning rate
    returns:
    - mean_acc [tensor]: training accuracy
    - mean_loss [tensor]: training loss
    """
    accuracies = []
    losses = []
    for X, Y in train_dataset:
        with tf.GradientTape() as tape:
            # forward pass
            X = X / 255.0
            y_hat = model(X)
            # calculate loss
            one_hot = tf.one_hot(Y, 43)
            loss = cross_entropy(y_hat, one_hot)
            losses.append(tf.math.reduce_mean(loss))

            grads = tape.gradient(loss, [W, b])
            sgd([W, b], grads, lr, X.shape[0]) 

            acc = accuracy(y_hat, Y)
            accuracies.append(acc)
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    mean_loss = tf.math.reduce_mean(losses)
    return mean_loss, mean_acc

def model(X):
    """
    logistic regression model
    """
    flatten_X = tf.reshape(X, (-1, W.shape[0]))
    return softmax(tf.matmul(flatten_X, W) + b)

    
def validation_loop():
    """
    loop through the validation dataset
    """
    accuracies = []
    for X, Y in val_dataset:
        X = X / 255.0
        y_hat = model(X)
        acc = accuracy(y_hat, Y)
        accuracies.append(acc)
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    return mean_acc


def get_module_logger(mod_name):

   # logging.getLogger --> returns a reference to a logger instance with
   # the specified name if provided, or root if not.
    logger = logging.getLogger(mod_name)

    # Returns a new instance of the StreamHandler class. If stream is specified, the instance will
    # use it for logging output; otherwise, sys.stderr will be used.
    handler = logging.StreamHandler()

    # For example: logging.Formatter('%(asctime)s - %(message)s', style='{').
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

    # setFormatter() is a Handler class method. it is used to format the log message.
    # all handlers use it for formatting.
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Sets the threshold for this logger to level.
    # Logging messages which are less severe than level will be ignored
    logger.setLevel(logging.DEBUG)

    return logger


if __name__ == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--imdir', default="../GTSRB/Final_Training/Images/", type=str,
                        help='data directory')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()

    #logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    logger.info('Training for {} epochs using {} data'.format(args.epochs,args.imdir))
    # get the datasets
    train_dataset, val_dataset = get_datasets(args.imdir)

    # set the variables
    num_inputs = 1024*3
    num_outputs = 43
    W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                    mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))
    
    lr = 0.1
    # training! 
    for epoch in range(args.epochs):
        #sys.stdout.write(f'Epoch {epoch}')
        logger.info('Epoch {}'.format(epoch))
        loss, acc = training_loop(lr)
        #sys.stdout.write(f'Mean training loss: {loss:1f}, mean training accuracy {acc:1f}')
        logger.info('Mean training loss: {:1f}, mean training accuracy {:1f}'.format(loss,acc))
        val_acc = validation_loop()
        #sys.stdout.write(f'Mean validation accuracy {val_acc:1f}')
        logger.info('Mean validation accuracy {:1f}'.format(val_acc))