import numpy as np
from activation_function import softmax
from loss import cross_entropy_loss

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # CSEloss
        self.y = None     # output of softmax
        self.t = None     # target vector (one-hot encoded)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # when the target is one-hot encoded
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
