from typing import Optional
import numpy as np
try:
    import cupy as cp
    xp = cp if cp.cuda.runtime.getDeviceCount() > 0 else np
except ImportError:
    xp = np

# The hidden layers are every layer in this program - initially I had created input and output layer classes, but
# I realized those were pointless, as you could just directly plug in the data into a hidden layer without an input layer
# and the output layer was largely just a child class from the hidden layer class, which I just merged to simplify things
class HiddenLayer:
    def __init__(self, n_count):
        self.prev = None
        self.n_count = n_count

        self.weights = None
        self.biases = None
# The weighted sum is the weights multiplied by outputs of previous layer with biases added. This is done by performing
# a matmul on arrays of the previous layers output and the current layers weights.
        self.weighted_sum = None
# The activated sum is a transformation of the weighted sum, such as relu or sigmoid, or softmax if used as the output layer
        self.activated_sum = None
# I initially stored gradients for weights and input, but realized I could just call the data directly during backprop without wasting memory
        self.act_grads = None
# The deltas represent the gradients with respect to loss
        self.w_deltas = None
        self.b_deltas = None
        self.i_deltas = None

    def xavier_init(self, prev):
        limit = xp.sqrt(6 / (prev + self.n_count))
        return xp.random.uniform(-limit, limit, size=(prev, self.n_count))

    def he_init(self, prev):
        return xp.random.normal(0, xp.sqrt(2 / prev), size=(prev, self.n_count))
# This initializes weights, and while you shouldn't need to use this if you just use the trainer class as intentioned, if you run this
# without the trainer class, you will need to initialize parameters before any further operations proceed
    def init_params(self, method, prev):
        if method == "xavier":
            self.weights = self.xavier_init(prev)
            self.biases = xp.zeros(shape=(1, self.n_count))
        elif method == "he":
            self.weights = self.he_init(prev)
            self.biases = xp.zeros(shape=(1, self.n_count))
        return self
# This prepares the weighted sum using matmul
    def hl_prep(self, matrix):
        self.weighted_sum = (matrix @ self.weights) + self.biases
        return self
# The relu activation function - ensure that 'training' mode is on during training, as that enables storing activation gradients
# These are not stored during inference, as it is a waste of compute and memory
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
# Only use softmax as the output layer, unless you are doing something more experimental - if that is the case, you will want to
# tweak the function, manually add gradients perhaps?
    def softmax(self):
        self.activated_sum = xp.exp(self.weighted_sum) / xp.sum(xp.exp(self.weighted_sum), axis=1, keepdims=True)
        return self