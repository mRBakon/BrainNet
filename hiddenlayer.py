from typing import Optional
import numpy as cp


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
        limit = cp.sqrt(6 / (prev + self.n_count))
        return cp.random.uniform(-limit, limit, size=(prev, self.n_count))

    def he_init(self, prev):
        return cp.random.normal(0, cp.sqrt(2 / prev), size=(prev, self.n_count))

    def init_params(self, method, prev):
        if method == "xavier":
            self.weights = self.xavier_init(prev)
            self.biases = cp.zeros(shape=(1, self.n_count))
        elif method == "he":
            self.weights = self.he_init(prev)
            self.biases = cp.zeros(shape=(1, self.n_count))
        return self

    def hl_prep(self, matrix):
        self.weighted_sum = (matrix @ self.weights) + self.biases
        return self

    def relu(self, mode, a):
        self.activated_sum = cp.maximum(self.weighted_sum, self.weighted_sum * -a)
        if mode == 'training':
            self.act_grads = (cp.where(self.activated_sum > 0, 1, -a))
        return self

    def sigmoid(self, mode):
        self.activated_sum = 1 / (1 + cp.exp(-self.weighted_sum))
        if mode == 'training':
            self.act_grads = (self.activated_sum * (1 - self.activated_sum))
        return self

    def softmax(self):
        self.activated_sum = cp.exp(self.weighted_sum) / cp.sum(cp.exp(self.weighted_sum), axis=1, keepdims=True)
        return self