import sys
import os
import numpy as np
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from network import TwoLayerNet
from common.optimizer import *

# load data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# hyper parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# init network
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
optimizer = Adam()

for i in range(iters_num):
    # mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grads = network.gradient(x_batch, t_batch)

    # gradient descent
    optimizer.update(network.params, grads)

    # calculate loss
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # accuracy for each epoch (96 ~ 97%)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
