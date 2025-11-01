import os
from hiddenlayer import HiddenLayer as HL
import numpy as cp
from sklearn.utils import shuffle


class Trainer:
    def __init__(self, hls: list, hla, ola, loss, lr):
        self.lr = lr

        self.hla = hla
        self.ola = ola
        self.loss = loss

        self.h_size = hls
        self.hl = []
        for h in hls:
            self.hl.append(HL(h))

        self.loss_time = []

    def initialize(self, method: str, data_size: int):
        self.hl[0].init_params(method, data_size)
        for h in range(1, len(self.hl)):
            self.hl[h].init_params(method, self.hl[h-1].weights.shape[1])

    def backprop(self, epoch_loss: list, data: cp.ndarray, target: cp.ndarray):
        if self.loss == 'mse':
            epoch_loss.append(cp.mean((target - self.hl[-1].activated_sum) ** 2))
            loss_grad = 2 * (self.hl[-1].activated_sum - target) / self.hl[-1].activated_sum.shape[0]
            self.hl[-1].b_deltas = loss_grad * self.hl[-1].act_grads
        elif self.loss == 'bce':
            epoch_loss.append(-cp.mean(target * cp.log(self.hl[-1].activated_sum) + (1 - target) * cp.log(1 - self.hl[-1].activated_sum)))
            loss_grad = ((1 - target) / (1 - self.hl[-1].activated_sum) - target / self.hl[-1].activated_sum) / self.hl[-1].activated_sum.shape[0]
            self.hl[-1].b_deltas = loss_grad * self.hl[-1].act_grads
        elif self.loss == 'ce':
            # epoch_loss.append((-1/target.shape[0]) * cp.sum(target * cp.log(self.hl[-1].activated_sum), axis=1))
            epoch_loss.append(-cp.mean(target * cp.log(self.hl[-1].activated_sum)))
            # bias deltas should always be a 1 by n matrix, as it is one bias per neuron, this also allows broadcasting across all matrices with equal n
            self.hl[-1].b_deltas = cp.mean(self.hl[-1].activated_sum - target, axis=0, keepdims=True)
        self.hl[-1].w_deltas = (self.hl[-1].b_deltas.T * cp.mean(self.hl[-2].activated_sum, axis=0, keepdims=True)).T
        self.hl[-1].i_deltas = cp.sum(self.hl[-1].b_deltas * self.hl[-1].weights, axis=1, keepdims=True)
        for h in reversed(range(1, len(self.hl) - 1)):
            self.hl[h].b_deltas = self.hl[h+1].i_deltas.T * cp.mean(self.hl[h].act_grads, axis=0, keepdims=True)
            self.hl[h].w_deltas = self.hl[h].b_deltas * cp.mean(self.hl[h-1].act_grads, axis=0, keepdims=True).T
            self.hl[h].i_deltas = cp.sum(self.hl[h].b_deltas * self.hl[h].weights, axis=1, keepdims=True)
        self.hl[0].b_deltas = self.hl[1].i_deltas.T * cp.mean(self.hl[0].act_grads, axis=0, keepdims=True)
        self.hl[0].w_deltas = self.hl[0].b_deltas * cp.mean(data, axis=0, keepdims=True).T
        for h in self.hl:
            h.weights -= self.lr * h.w_deltas
            h.biases -= self.lr * h.b_deltas

    def batch(self, epoch_loss: list, data: cp.ndarray, target: cp.ndarray):
        if self.hla == 'relu':
            self.hl[0].hl_prep(data).relu('training', a=0.001)
            for h in range(1, len(self.hl) - 1):
                self.hl[h].hl_prep(self.hl[h-1].activated_sum).relu('training', a=0.01)
        elif self.hla == 'sigmoid':
            self.hl[0].hl_prep(data).sigmoid('training')
            for h in range(1, len(self.hl) - 1):
                self.hl[h].hl_prep(self.hl[h-1].activated_sum).sigmoid('training')
        if self.ola == 'sigmoid':
            self.hl[-1].hl_prep(self.hl[-2].activated_sum).sigmoid('training')
        elif self.ola == 'softmax':
            self.hl[-1].hl_prep(self.hl[-2].activated_sum).softmax()
        self.backprop(epoch_loss, data, target)

    def epoch(self, n_batches, s_d: cp.ndarray, s_t: cp.ndarray):
        epoch_loss = []
        for b in range(n_batches):
            mini_data = s_d[b * len(s_d)//n_batches:(b + 1) * len(s_d)//n_batches]
            mini_target = s_t[b * len(s_t)//n_batches:(b + 1) * len(s_t)//n_batches]
            self.batch(epoch_loss, mini_data, mini_target)
        self.loss_time.append(cp.average(epoch_loss))

    def train(self, n_epochs, n_batches, data, target):
        for epoch in range(n_epochs):
            s_d = shuffle(data, random_state=42)
            s_t = shuffle(target, random_state=42)
            self.epoch(n_batches, s_d, s_t)
        return self

    def save(self, name: str):
        lib = {"HLA": self.hla, "OLA": self.ola}
        for i, layer in enumerate(self.hl):
            lib[f'layer{i}_W'] = layer.weights
            lib[f'layer{i}_B'] = layer.biases
        b_dir = os.path.dirname(os.path.abspath(__file__))
        m_dir = os.path.join(b_dir, "Models")
        os.makedirs(m_dir, exist_ok=True)
        path = os.path.join(m_dir, name)
        cp.savez(path, **lib)
        return self






