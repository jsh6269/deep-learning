import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        # f(x+h)
        x[idx] = tmp + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp - h
        fxh2 = f(x)

        # df/dx
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp  # restore

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)
