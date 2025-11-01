from typing import Optional
import numpy as np
try:
    import cupy as cp
    xp = cp if cp.cuda.runtime.getDeviceCount() > 0 else np
except ImportError:
    xp = np


class HiddenLayer:
    def __init__(self, n_count):
        self.prev = None
        self.n_count = n_count

        self.weights = None
        self.biases = None

        self.weighted_sum = None
        self.activated_sum = None

        self.act_grads = None

        self.w_deltas = None
        self.b_deltas = None
        self.i_deltas = None

    def xavier_init(self, prev):
        limit = xp.sqrt(6 / (prev + self.n_count))
        return xp.random.uniform(-limit, limit, size=(prev, self.n_count))

    def he_init(self, prev):
        return xp.random.normal(0, xp.sqrt(2 / prev), size=(prev, self.n_count))

    def init_params(self, method, prev):
        if method == "xavier":
            self.weights = self.xavier_init(prev)
            self.biases = xp.zeros(shape=(1, self.n_count))
        elif method == "he":
            self.weights = self.he_init(prev)
            self.biases = xp.zeros(shape=(1, self.n_count))
        return self

    def hl_prep(self, matrix):
        self.weighted_sum = (matrix @ self.weights) + self.biases
        return self

    def relu(self, mode, a):
        self.activated_sum = xp.maximum(self.weighted_sum, self.weighted_sum * -a)
        if mode == 'training':
            self.act_grads = (xp.where(self.activated_sum > 0, 1, -a))
        return self

    def sigmoid(self, mode):
        self.activated_sum = 1 / (1 + xp.exp(-self.weighted_sum))
        if mode == 'training':
            self.act_grads = (self.activated_sum * (1 - self.activated_sum))
        return self

    def softmax(self):
        self.activated_sum = xp.exp(self.weighted_sum) / xp.sum(xp.exp(self.weighted_sum), axis=1, keepdims=True)
        return self