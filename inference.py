import subprocess
comp_gpu = None
try:
    subprocess.check_output('nvidia-smi')
    comp_gpu = True
except Exception:
    comp_gpu = False

import numpy as np
try:
    import cupy as xp
    xp = xp if xp.cuda.runtime.getDeviceCount() > 0 else np
except ImportError:
    xp = np
from hiddenlayer import HiddenLayer as HL

# This is the class used to run AI models that you have created with the trainer class
# You need to include the directory to the .npz file with model parameters, as well as specify what activation functions you used
class Inference:
    def __init__(self, model: str, hla: str, ola: str):
        self.m_dir = model
        self.hla = hla
        self.ola = ola
        # The hl is short for hidden layer, and will contain class instances of Hidden Layers
        self.hl = []

    # Loads in the model parameters
    def initialize(self):
        i = 0
        data = np.load(self.m_dir)
        if comp_gpu:
            data_gpu = {k: xp.array(v) for k, v in data.items()}
            while f"layer{i}_W" in data:
                weights = data_gpu[f"layer{i}_W"]
                biases = data_gpu[f"layer{i}_B"]
                self.hl.append(HL(n_count=weights.shape[1]))
                self.hl[i].weights = weights
                self.hl[i].biases = biases
                i += 1
        elif comp_gpu:
            while f"layer{i}_W" in data:
                weights = data[f"layer{i}_W"]
                biases = data[f"layer{i}_B"]
                self.hl.append(HL(n_count=weights.shape[1]))
                self.hl[i].weights = weights
                self.hl[i].biases = biases
                i += 1
        return self

    # This runs the model on your data - consider increasing the "data" size you use for batching, you can get the results for numerous inputs at once
    # However, if you don't want to do that, a for loop could work
    def run(self, data: xp.ndarray):
        if self.hla == 'sigmoid':
            # Technically speaking, you don't need to set the mode to "inference", in fact, it would still work if you typed in "training"
            # I just don't want it to waste time calculating gradients when it isn't learning. Since the code only uses an if statement to check
            # if mode == "training", you could choose any word whatsoever
            self.hl[0].hl_prep(data).sigmoid(mode='inference')
            for h in range(1, len(self.hl) - 1):
                self.hl[h].hl_prep(self.hl[h-1].activated_sum).sigmoid(mode='inference')
        elif self.hla == 'relu':
            self.hl[0].hl_prep(data).relu()
            for h in range(1, len(self.hl) - 1):
                # To my understanding, the "a" parameter is somewhat arbitrary - you can try fiddling with it, but I imagine lower values work better so the gradients don't explode
                self.hl[h].hl_prep(self.hl[h-1].activated_sum).relu(a=0.01)
        if self.ola == 'sigmoid':
            self.hl[-1].hl_prep(self.hl[-2].activated_sum).sigmoid(mode='inference')
        elif self.ola == 'softmax':
            self.hl[-1].hl_prep(self.hl[-2].activated_sum).softmax()
        return self.hl[-1].activated_sum