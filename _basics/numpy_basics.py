import numpy as np

# initialize numpy array
x = np.array([1.0, 2.0, 3.0])
y = np.asarray([3.0, 4.0, 1.0])

print(x, y)
print(type(x))

a1 = np.zeros((2, 3)) # [[0, 0, 0], [0, 0, 0]]
a2 = np.ones((2, 3)) # [[1, 1, 1], [1, 1, 1]]
a3 = np.full((2, 3), 1.2) # [[1.2, 1.2, 1.2], [1.2, 1.2, 1.2]]
i3 = np.identity(3) # identity matrix w/ size (3, 3)

e1 = np.empty((2, 3)) # not initialized

# array with same size
e2 = np.empty_like(e1)
ol = np.ones_like(e1)
zl = np.zeros_like(e1)

# array with random number
n1 = np.random.rand(2, 3) # uniform distribution from [0, 1]
n2 = np.random.randn(2, 3) # N(0, 1)
n3 = np.random.randint(1, 10, (2, 3)) # random int from range [1, 10]

# component-wise operation
print(x + y)
print(x * y)

z = np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]])
w = np.array([[2, 1, 1], [3, 2, 1], [5, 1, 2]])
k = np.array([[1, 1, 1], [2, 2, 2]])
print(z)
print(k.shape) # (2, 3)

# transpose
print(z.T)

# scalar multiplication
print(z * 3)

# matrix multiplication
print(z @ w)
print(np.dot(z, w))

# component-wise comparison
# (same size as z)
print(z > 2)

# filtering
s = np.array([[1, 2, 3], [4, 5, 6]])
indices = np.where(s > 2)
print(indices)  # (array([0, 1, 1, 1]), array([2, 0, 1, 2]))
print(s[indices])  # [3 4 5 6]

# min, max
zp = np.array([[1, 2, 3], [7, 8, 9]])
print(np.argmax(zp))  # max index (5)
print(np.argmax(zp, axis=0))  # max index from each column [1 1 1]
print(np.argmax(zp, axis=1))  # max index from each row [2 2]

# matrix flattening
# [1, 2, 3, 4, 5, 6, 4, 2, 1]
u1 = z.flatten() # independent from z
u2 = z.ravel() # shares memory w/ z

z[0][0] = -1
print(u1)
print(u2)

u2[0] = 1
print(z)

# shape conversion
# [[1], [1], [1], [2], [2], [2]]
t = k.reshape(6, 1)
print(t)

# direct shape conversion
# [[1, 1], [1, 2], [2, 2]]
k.shape = (3, 2)
print(k)

# copy & view
# view shares memory w/ the original
# however properties such as shape, type are independent!
# type = interpretation (does not affect the data itself)
z = np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]], dtype=np.int16)
q = z.copy()
v = z.view(np.int8)

v[0][0] = 5
v.shape = (1, 9 * 2)
print(v) # [[5 0 2 0 3 0 4 0 5 0 6 0 4 0 2 0 1 0]]
print(z) # [[5, 2, 3], [4, 5, 6], [4, 2, 1]]
print(q) # [[1, 2, 3], [4, 5, 6], [4, 2, 1]]

# type conversion
print(q.astype(float))

# vstack: stack vertically
# hstack: stack horizontally
a = np.array([[1, 2], [3, 4]])

'''
[[1 2]
 [3 4]
 [5 6]]
'''
b = np.array([[5, 6]])
print(np.vstack((a, b)))
print(np.concatenate((a, b), axis=0)) # the same as above

'''
[[1 2 5]
 [3 4 6]]
'''
c = np.array([[5], [6]])
print(np.hstack((a, c)))
print(np.concatenate((a, c), axis=1)) # the same as above
