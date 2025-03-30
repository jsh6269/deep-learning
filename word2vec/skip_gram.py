import sys
sys.path.append('..')
import numpy as np
from _common.matmul import MatMul
from _common.softmax import SoftmaxWithLoss


class SkipGram:
    def __init__(self, vocab_size, hidden_size, window_size=2):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer = MatMul(W_in)
        self.out_layers = [MatMul(W_out) for _ in range(window_size)]
        self.loss_layers = [SoftmaxWithLoss() for _ in range(window_size)]

        # 모든 가중치와 기울기를 리스트에 모음
        self.params, self.grads = [], []
        layers = [self.in_layer] + self.out_layers
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어 벡터 저장
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        loss = 0
        for i, (out_layer, loss_layer) in enumerate(zip(self.out_layers, self.loss_layers)):
            s = out_layer.forward(h)
            loss += loss_layer.forward(s, contexts[:, i])
        return loss

    def backward(self, dout=1):
        dh = 0
        for out_layer, loss_layer in zip(self.out_layers, self.loss_layers):
            ds = loss_layer.backward(dout)
            dh += out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
