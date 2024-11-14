from collections import OrderedDict
import numpy as np
from errors import Errors
from activations import Activations
from layers import Relu, Sigmoid, Affine, SoftmaxWithLoss
from tqdm import tqdm
import pickle

class TwoLayerNetWithBackProp:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.params['w1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['w2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.activations = Activations()
        self.errors = Errors()

        # create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.last_layer = SoftmaxWithLoss()
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []



    def predict(self, x):
        """
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, w1) + b1
        z1 = self.activations.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = self.activations.softmax(a2)
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        y = x

        return y
    
    def loss(self, x, y):
        y_hat = self.predict(x)
        
        # note: different return value
        return self.last_layer.forward(y_hat, y)
    

    def accuracy(self, x, y):
        y_hat = self.predict(x)
        p = np.argmax(y_hat, axis=1)
        y_p = np.argmax(y, axis=1)

        return np.sum(p == y_p)/float(x.shape[0])
    

    # for multi-dimensional x
    def _numerical_gradient(self, f, x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
        
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x) # f(x+h)
            
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            
            x[idx] = tmp_val 
            it.iternext()   
            
        return grad
    

    def numerical_gradient(self, x, y):
        loss_w = lambda w: self.loss(x, y)

        grads = {}
        grads['w1'] = self._numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = self._numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = self._numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = self._numerical_gradient(loss_w, self.params['b2'])

        return grads
    
    def gradient(self, x, y):
        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads
    
    def fit(self, iterations, x_train, t_train, x_test, t_test, \
            batch_size, learning_rate=0.1, backprop=True):
        
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size/batch_size, 1)
        
        print('Start training......')
        
        for i in tqdm(range(iterations)):
            # get mini batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            #calculate slopes (gradients)
            if backprop:
                grad = self.gradient(x_batch, t_batch)
            else:
                grad = self.numerical_gradient(x_batch, t_batch)
                
            for key in ('w1', 'b1', 'w2', 'b2'):
                self.params[key] -= learning_rate * grad[key]
                
            loss = self.loss(x_batch, t_batch)
            
            self.train_losses.append(loss)
            
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                self.train_accs.append(train_acc)
                self.test_accs.append(test_acc)
            
        print('Done.')

    def save_model(self, model_filename):
        print('Saving model....')
        with open(model_filename, 'wb') as f:
            pickle.dump(self.params, f, -1)
        print('Done saving model')

    def load_model(self, model_filename):
        print('Loading model.....')
        with open(model_filename, 'rb') as f:
            self.params = pickle.load(f)
        print('Done')
        

        