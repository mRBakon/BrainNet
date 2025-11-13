import os
from hiddenlayer import HiddenLayer as HL
import numpy as np
try:
    import cupy as cp
    xp = cp if cp.cuda.runtime.getDeviceCount() > 0 else np
except ImportError:
    xp = np
    
from sklearn.utils import shuffle

# The brunt of the work, the "Trainer" class trains Neural Networks given information such as the size of the network,
# what activation functions, what loss function, what learning rate, and later on, data is plugged in. At the beginning,
# all you need to plug in is a list of integers - the length of the list is the number of layers, and the value of each number
# is the number of "neurons" per layer
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

    # Absolutely crucial to perform this step ONCE at the beginning - otherwise it will try performing math with numbers that don't exist
    # which will produce errors. For method, choose Xavier or He as methods, and for data size, choose the size of an input sample
    # for instance, in the iris dataset, a single sample gives you 4 datapoints to work off of, so for any problem using the iris dataset
    # you would set the datasize to 4
    def initialize(self, method: str, data_size: int):
        self.hl[0].init_params(method, data_size)
        for h in range(1, len(self.hl)):
            self.hl[h].init_params(method, self.hl[h-1].weights.shape[1])

    # This is the most important part of the program - it calculates the gradients and updates the weights - any learning going on occurs here
    # At the moment I have only implemented naive batch SGD, but may eventually add momentum, perhaps up to Nesterov momentum if I can
    def backprop(self, epoch_loss: list, data: xp.ndarray, target: xp.ndarray):
        # MSE networks have not been tested to work, and should not be used until further notice.
        if self.loss == 'mse':
            epoch_loss.append(xp.mean((target - self.hl[-1].activated_sum) ** 2))
            loss_grad = 2 * (self.hl[-1].activated_sum - target) / self.hl[-1].activated_sum.shape[0]
            self.hl[-1].b_deltas = xp.mean(loss_grad * self.hl[-1].act_grads, axis=0, keepdims=True)
            # Binary cross entropy networks using sigmoid do not currently work (produces output without errors, but no learning),
            # and this should not be used until it has been fixed.
        elif self.loss == 'bce':
            epoch_loss.append(-xp.mean(target * xp.log(self.hl[-1].activated_sum) + (1 - target) * xp.log(1 - self.hl[-1].activated_sum)))
            loss_grad = ((1 - target) / (1 - self.hl[-1].activated_sum) - target / self.hl[-1].activated_sum) / self.hl[-1].activated_sum.shape[0]
            self.hl[-1].b_deltas = xp.mean(loss_grad * self.hl[-1].act_grads, axis=0, keepdims=True)
        elif self.loss == 'ce':
            epoch_loss.append(-xp.mean(target * xp.log(self.hl[-1].activated_sum)))
            self.hl[-1].b_deltas = xp.mean(self.hl[-1].activated_sum - target, axis=0, keepdims=True)
        self.hl[-1].w_deltas = (self.hl[-1].b_deltas.T * xp.mean(self.hl[-2].activated_sum, axis=0, keepdims=True)).T
        self.hl[-1].i_deltas = xp.sum(self.hl[-1].b_deltas * self.hl[-1].weights, axis=1, keepdims=True)
        for h in reversed(range(1, len(self.hl) - 1)):
            self.hl[h].b_deltas = self.hl[h+1].i_deltas.T * xp.mean(self.hl[h].act_grads, axis=0, keepdims=True)
            self.hl[h].w_deltas = self.hl[h].b_deltas * xp.mean(self.hl[h-1].act_grads, axis=0, keepdims=True).T
            self.hl[h].i_deltas = xp.sum(self.hl[h].b_deltas * self.hl[h].weights, axis=1, keepdims=True)
        self.hl[0].b_deltas = self.hl[1].i_deltas.T * xp.mean(self.hl[0].act_grads, axis=0, keepdims=True)
        self.hl[0].w_deltas = self.hl[0].b_deltas * xp.mean(data, axis=0, keepdims=True).T
        for h in self.hl:
            h.weights -= self.lr * h.w_deltas
            h.biases -= self.lr * h.b_deltas

    #
    def batch(self, epoch_loss: list, data: xp.ndarray, target: xp.ndarray):
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

    def epoch(self, n_batches, s_d: xp.ndarray, s_t: xp.ndarray):
        epoch_loss = []
        for b in range(n_batches):
            mini_data = s_d[b * len(s_d)//n_batches:(b + 1) * len(s_d)//n_batches]
            mini_target = s_t[b * len(s_t)//n_batches:(b + 1) * len(s_t)//n_batches]
            self.batch(epoch_loss, mini_data, mini_target)
        self.loss_time.append(xp.average(xp.asarray(epoch_loss)).item())

    def train(self, n_epochs, n_batches, data, target):
        for epoch in range(n_epochs):
            s_d = shuffle(data, random_state=42)
            s_t = shuffle(target, random_state=42)
            self.epoch(n_batches, s_d, s_t)
        return self

    def save(self, name: str):
        lib = {}
        for i, layer in enumerate(self.hl):
            lib[f'layer{i}_W'] = layer.weights
            lib[f'layer{i}_B'] = layer.biases
        b_dir = os.path.dirname(os.path.abspath(__file__))
        m_dir = os.path.join(b_dir, "Models")
        os.makedirs(m_dir, exist_ok=True)
        path = os.path.join(m_dir, name)
        xp.savez(path, **lib)
        return self






