import numpy as np
from scipy.special import expit

def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache
    
def affine_sig_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, sig_cache = sig_forward(a)
    cache = (fc_cache, sig_cache)
    return out, cache

def affine_forward(x, w, b):
    out = None
    N = x.shape[0]
    x_temp = x.reshape(N,-1)
    out = x_temp.dot(w) + b
    cache = (x, w, b)
    return out, cache
    

def relu_forward(x):
    out = None
    out = np.maximum(0,x)
    cache = x
    return out, cache
    
def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0
    return dx
    
    
def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    x_r = x.reshape(x.shape[0], w.shape[0])
    dw = np.dot(x_r.T,dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    return dx, dw, db
    
def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
    
def affine_sig_backward(dout, cache):
    fc_cache, sig_cache = cache
    da = (1-dout)*dout
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

    
def sig_forward(x):
    out = None
    out = expit(x)
    cache = x
    return out, cache
    
def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx