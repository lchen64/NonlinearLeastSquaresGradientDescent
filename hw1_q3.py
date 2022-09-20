# Imports
import numpy as np
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
import numdifftools as nd

class nonlinear_ls_grad_descent(object):
  """ Homework 1 Question 3 """

  def eval_L1(self, x):
    """Evaluate L(x)=(1-x1)^2+200*(x2-x1^2)^2"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: L (numpy.float64)"""

    L = (1-x[0])**2 + 200*(x[1]-(x[0]**2))**2
    
    return L


  def eval_G1(self, x):
    """Evaluate gradient of L(x)=(1-x1)^2+200*(x2-x1^2)"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: G (numpy.ndarray, size: 2)"""
    
    G1 = -398*x[0]
    G2 = 200
    G = np.array([G1, G2])
    
    return G


  def eval_H1(self, x):
    """Evaluate hessian of L(x)=(1-x1)^2+200*(x2-x1^2)"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: H (numpy.ndarray, size: 2x2)"""

    H = np.zeros((2, 2))
    H[0][0] = -800*(x[1]-x[0]**2)+1600*x[0]**2+2
    H[0][1] = -800*x[0]
    H[1][0] = -800*x[0]
    H[1][1] = 400
    
    return H

  def gradient_descent(self, x0, fx, gx):
    """Perform gradient descent"""
    """Inputs: x0 (numpy.ndarray, size: 2), 
               fx (loss function), 
               gx (gradient function)"""
    """Outputs: xs (numpy.ndarray, size: (numSteps+1)x2)"""
    
    learn_rate = .2
    x = x0.astype(float) #initial starting point 
    
    xs = []
    for i in range(500000):
        diff = -learn_rate * gx(x)
        x += diff
        xs.append(x)
        
    return np.array(xs)


  def newton_descent(self, x0, fx, gx, hx):
    """Perform gradient descent"""
    """Inputs: x0 (numpy.ndarray, size: 2), 
               fx (loss function), 
               gx (gradient function)
               hx (hessian function)"""
    """Outputs: xs (numpy.ndarray, size: (numSteps+1)x2)"""

    learn_rate = .2
    
    x = x0.astype(float) #initial starting vector
    
    xs = []
    for i in range(500000):
        
        diff = -learn_rate * numpy.matmul(numpy.linalg.inv(hx(x)),gx(x))
        x += diff
        xs.append(x)
        
    return np.array(xs)


  def eval_L2(self, x):
    """Evaluate L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: L (numpy.float64)"""
    L = x[0]*exp((-x[0]**2)-(1/2)*x[1]**2) + x[0]**2/10 + x[1]**2/10
    return L


  def eval_G2(self, x):
    """Evaluate gradient of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: G (numpy.ndarray, size: 2)"""
    
    g = lambda x1,x2: x1*exp(-x1**2-(1/2)*x2**2) + x1**2/10 + x2**2/10
    G = nd.Gradient(g)([x[0],x[1]])
    return G


  def eval_H2(self, x):
    """Evaluate hessian of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: H (numpy.ndarray, size: 2x2)"""
    
    h = lambda x1,x2: x1*exp(-x1**2-(1/2)*x2**2) + x1**2/10 + x2**2/10
    H = nd.Hessian(h)([x[0],x[1]])
    return H


if __name__ == '__main__':
    
    """This code runs if you execute this script"""
    nonlinear_ls_grad_descent = nonlinear_ls_grad_descent()

    # # TODO: Uncomment the following code to visualize gradient descent
    # #       & newton's method for the first loss function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = np.arange(-2, 2, 0.01)
    x2 = np.arange(-2, 2, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
         for j in range(X1.shape[1]):
            Z[i, j] = hw1_q3.eval_L1(np.array([X1[i, j], X2[i, j]]))
    contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    xsg = hw1_q3.gradient_descent(np.array([0, 0]), hw1_q3.eval_L1, 
         hw1_q3.eval_G1)
    xsn = hw1_q3.newton_descent(np.array([0, 0]), hw1_q3.eval_L1, 
         hw1_q3.eval_G1, hw1_q3.eval_H1)
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("First loss function gradient method steps: ", xsg.shape[0]-1)
    print("First loss function newton method steps:   ", xsn.shape[0]-1)


    # # TODO: Uncomment the following code to visualize gradient descent
    # #       & newton's method for the second loss function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = np.arange(-3.5, 3.5, 0.01)
    x2 = np.arange(-3.5, 3.5, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
         for j in range(X1.shape[1]):
            Z[i, j] = hw1_q3.eval_L2(np.array([X1[i, j], X2[i, j]]))
    contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    xsg = hw1_q3.gradient_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2, 
        hw1_q3.eval_G2)
    print("-"*50)
    xsn = hw1_q3.newton_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2, 
    hw1_q3.eval_G2, hw1_q3.eval_H2) # Not quite working yet
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("Second loss function gradient method steps: ", xsg.shape[0]-1)
    print("Second loss function newton method steps:   ", xsn.shape[0]-1)
    plt.show()