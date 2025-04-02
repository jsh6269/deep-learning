import sys
sys.path.append('..')
import numpy as np
from GRU.gru import TimeGRU


# 배치 크기, 시퀀스 길이, 입력 차원, 은닉 상태 크기
N, T, D, H = 20, 100, 25, 8
xs = np.random.randn(N, T, D)
h0 = np.zeros((N, H))

# Xavier init
Wx = np.random.randn(D, 3 * H) * np.sqrt(1.0 / D)
Wh = np.random.randn(H, 3 * H) * np.sqrt(1.0 / H)

gru = TimeGRU(Wx, Wh, stateful=False)

learning_rate = 0.01
epochs = 2000

# target: sin
time_steps = np.linspace(0, np.pi, T)
fixed_target = np.sin(time_steps)[None, :, None]
target = np.tile(fixed_target, (N, 1, H))

for epoch in range(epochs):
  hs = gru.forward(xs)
  
  # MSE Loss
  loss = np.mean((hs - target) ** 2)
  
  # backprop
  dhs = 2 * (hs - target) / np.prod(hs.shape)
  dxs = gru.backward(dhs)
  
  # update weights
  gru.Wx -= learning_rate * gru.dWx
  gru.Wh -= learning_rate * gru.dWh
  
  if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
