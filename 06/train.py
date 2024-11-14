
import mnist
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

import matplotlib.pyplot as plt
import numpy as np

#const definition
mnist_pkl_filename = 'pawar_mnist_model.pkl'

#load mnist datasets
my_mnist = mnist.Mnist()
(x_train, t_train), (x_test, t_test) = my_mnist.load()
                                        

#hyperparameters

iterations = 10000
batch_size = 16
learning_rate = 0.01

network = TwoLayerNetWithBackProp(input_size = 28*28, hidden_size = 100, output_size = 10)

network.fit(iterations, x_train, t_train, x_test, t_test, batch_size, learning_rate=learning_rate,
            backprop=True)

network.save_model(mnist_pkl_filename)