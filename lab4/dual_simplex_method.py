import numpy as np
from main_simplex import get_c_b, get_A_b_matrix
import math


# c_vector = np.matrix([
#   [-4],
#   [-3],
#   [-7],
#   [0],
#   [0]
# ])
# A_matrix = np.matrix([
#  [-2, -1, -4, 1, 0],
#  [-2, -2, -2, 0, 1]
# ])
# 
# b_vector = np.matrix([
#   [-1],
#   [-1.5]
# ])
# J_b = [3, 4]

c_vector = np.matrix([[2], [2], [1], [-10],  [1],  [4],  [-2], [-3]])

A_matrix = np.matrix([
  [-2, -1, 1,  -7, 0,  0,  0,  2],
  [4, 2,  1,  0,  1,  5,  -1, -5],
  [1, 1,  0,  -1, 0,  3,  -1, 1]
])

b_vector = np.matrix([[-2],[4],[3]])

J_b = [2, 5, 7]


def foreach_predicate(iterable, predicate):
    for i, item in enumerate(iterable):
        if not predicate(item):
            return i, item
    return -1, None

m, n = A_matrix.shape
J = [j for j in range(n)]

while (True):
  A_b_inv_matrix = get_A_b_matrix(A_matrix, J_b).I
  y_vector = (get_c_b(c_vector.A1, J_b) * A_b_inv_matrix).T

  kappa_b = (A_b_inv_matrix * b_vector).A1

  j_negative, negative_kappa_j = foreach_predicate(kappa_b, lambda v: v >= 0)

  kappa_vector = [ 
    kappa_b[J_b.index(j)] if (j in J_b) else 0 for j in J
  ]

  if (j_negative == -1):
    print("success")
    print(kappa_vector)
    print(c_vector.T*np.matrix(kappa_vector).T)
    break

  delta_y__t = A_b_inv_matrix[j_negative,:]

  mu = []
  min_ksi = math.inf
  min_ksi_index = -1
  has_negative_items = False
  for j in J:
    if j in J_b:
      mu.append(0)
      continue
    mu_j = delta_y__t * A_matrix[:,j]
    mu.append(mu_j)
    if mu_j < 0:
      has_negative_items = True

      ksi = (( c_vector.A1[j] - A_matrix[:,j].T * y_vector ) / mu_j)[0,0]
      if ksi < min_ksi:
        min_ksi = ksi
        min_ksi_index = j

  if not has_negative_items:
    print("not solvable")
    raise Exception("not solvable")

  J_b[j_negative] = min_ksi_index
  y_vector = y_vector - min_ksi * delta_y__t.T







