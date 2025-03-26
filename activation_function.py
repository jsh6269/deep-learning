import numpy as np

def step_function(x):
	y = x > 0
	return y.astype(np.int64)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
	return np.where(x > 0, x, alpha * x)

def tanh(x):
	return np.tanh(x)

def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T

	x = x - np.max(x) # prevent overflow
	return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
	a, b = np.array([3]), np.array([1, -2, 3])

	for func in [step_function, sigmoid, relu, leaky_relu, tanh, softmax]:
		print(func(a))
		print(func(b))
