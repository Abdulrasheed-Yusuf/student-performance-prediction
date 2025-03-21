import numpy as np
import pandas as pd

def compute_cost(X, y, w, b):
    m, n = X.shape
    
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += np.square(f_wb_i - y[i])
        
    total_cost = cost / (2 * m)
    
    return total_cost

def compute_gradients(X, y, w, b):
    m, n = X.shape
    
    dj_dw = np.zeros(n)
    dj_db = 0
    
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += X[i, j] * error
        dj_db += error
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(X, y, alpha, num_iter=1000):
    m, n = X.shape
    
    w = np.zeros(n)
    b = 0
    
    cost_history = []
    
    for i in range(num_iter):
        dj_dw, dj_db = compute_gradients(X, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        cost = (1 / (2 * m)) * np.sum((np.dot(X, w) + b - y) ** 2)
        cost_history.append(cost)
        
    return w, b, cost_history


def predict(X, w, b):
    y_pred = np.dot(X, w) + b
    
    return y_pred
    
        