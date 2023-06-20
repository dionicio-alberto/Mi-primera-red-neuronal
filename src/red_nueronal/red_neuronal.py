import numpy as np
import math_funciones.math_funciones as mf

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(y.shape)
        self.prediction = 0
        
    def feedforward(self):
        self.layer1 = mf.sigmoid(np.dot(self.input, self.weights1))
        self.output = mf.sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * mf.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * mf.sigmoid_derivative(self.output), self.weights2.T) * mf.sigmoid_derivative(self.layer1)))
                            
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    def aplicar_modelo(self,x):
        self.layer1 = mf.sigmoid(np.dot(x, self.weights1))
        self.prediction = mf.sigmoid(np.dot(self.layer1, self.weights2))