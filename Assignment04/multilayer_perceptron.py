import numpy as np

class MultilayerPerceptron:
    def __init__(self, w1, b1, w2, b2, w3, b3):
        self.net = {}
        
        self.net['w1'] = w1
        self.net['b1'] = b1
        
        self.net['w2'] = w2
        self.net['b2'] = b2
        
        self.net['w3'] = w3
        self.net['b3'] = b3
        
    def sigmoid(self, a):
        return 1/(1 + np.exp(-a))
    
    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']
        
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)
        
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        
        a3 = np.dot(z2, w3) + b3
        
        return a3

if __name__ == "__main__":
    
    print("Multilayer Perceptron class calculates output using a three-layer neural network.")
    x = np.array([0.5, 1]) 
    print("\n\nNeural Network Input")
    print(x)
    
    print("Set of Weights and Biases to calculate second layer nodes")
    w1 = np.array([[0.1, 0.2, 0.15], [0.6, 0.2, 0.8]])
    b1 = np.array([0.1, 0.2, 0.05])
    print("Weights w1:- ", w1)
    print("Bias b1:- ", b1)
    
    print("Set of Weights and Bias to calculate third layer nodes.")
    w2 = np.array([[0.1, 0.2], [0.05, 0.5], [0.1, 0.2]])
    b2 = np.array([0.4, 0.5])
    print("Weight w2:- ", w2)
    print("Wias b2:- ", b2)
    
    print("Weights and bias used to calculate the output nodes.")
    w3 = np.array([[0.1, 0.2], [0.9, 0.8]])
    b3 = np.array([0.9, 0.9])
    print("Weight w3:- ",w3)
    print("Bias b3:- ",b3)
    
    print("Forward function calculates output using Weights and Biases.")
    print("\n\nExample: Level 1 node takes inputs as numpy arrays.")
    print("Layer 0 inputs include Weights (w1, w2, w3) and Biases (b1, b2, b3).")
    print("These produce a1 at Layer 1, followed by activation to get z1.")
    print("z1 is used to calculate z2 at Layer 2 using sigmoid activation on a2.")
    print("Finally, z2 and the third set of Weights and Biases give the output.")

    y = MultilayerPerceptron(w1, b1, w2, b2, w3, b3)
    z = y.forward(x)
    print("\n\nOutput OF above multilayer neural network is: ",z)
    
    
    
    
        
        