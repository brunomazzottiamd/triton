import numpy as np

A = np.array([
    [-3, -2, -5],
    [-4, -3, 4],
    [-1, 0, -3],
    [5, 2, -1],
], dtype=np.int8, order="C")
# A = np.array([
#     [-3, -4, 1],
#     [5, -2, -3],
#     [0, 2, -5],
#     [4, -3, -2],
# ], dtype=np.int8, order="C")
print("A = \n", A)
print("shape = ", A.shape)
print("strides = ", A.strides)

B = np.array([
    [3, 2, 5, -1, -4],
    [-4, 4, -3, -1, -2],
    [-4, -5, 1, -1, 3],
], dtype=np.int8, order="F")
print("B = \n", B)
print("shape = ", B.shape)
print("strides = ", B.strides)

C = A @ B
print("C = \n", C)
print("shape = ", C.shape)
print("strides = ", C.strides)
