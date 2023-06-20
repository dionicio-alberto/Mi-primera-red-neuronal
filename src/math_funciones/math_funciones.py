import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def square_error(y1,y2):
    s_e = (y1-y2)**2
    return s_e

def sum_s_e(y1,y2):
    if len(y1)!=len(y2):
        return print('Error, dimensions aren´t equal')
    else:
        n=len(y1)
        sum = 0
        for i in range(n):
            sum += square_error(y1[i],y2[i])
        return sum
    
def sigmoid_derivative(x):
     sd=sigmoid(x)*(1-sigmoid(x))
     return sd
