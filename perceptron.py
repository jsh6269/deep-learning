import numpy as np

def perceptron(x1, x2, w1, w2, b):
  x = np.array([x1, x2])
  w = np.array([w1, w2])
  if np.sum(w * x) + b <= 0:
    return 0
  else:
    return 1

def AND(x1, x2):
  return perceptron(x1, x2, 0.5, 0.5, -0.7)

def NAND(x1, x2):
  return perceptron(x1, x2, -0.5, -0.5, 0.7)

def OR(x1, x2):
  return perceptron(x1, x2, 0.5, 0.5, -0.2)

def NOR(x1, x2):
  return perceptron(x1, x2, -0.5, -0.5, 0.2)

def NOT(x):
  return perceptron(x, 0, -1, 0, 0.5)

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

def XNOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = NAND(s1, s2)
  return y

def EQUAL(x1, x2):
  return XNOR(x1, x2)
