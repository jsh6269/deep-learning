import numpy as np

def cross_entropy_loss(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def mean_squared_error(y, t):
    return np.mean((y - t) ** 2)

def huber_loss(y, t, delta=1.0):
    error = y - t
    condition = np.abs(error) <= delta
    return np.mean(np.where(condition, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta)))

def cosine_similarity_loss(y, t):
    y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-7)
    t_norm = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-7)
    return -np.sum(y_norm * t_norm, axis=-1)
