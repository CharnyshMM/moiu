import numpy as np


def get_l_vector(a_inv_mtrx, x_vect):
    return a_inv_mtrx * x_vect

def get_Q_matrix(l_vect, i, n):
    l_tilda_vector = l_vect.copy()
    l_tilda_vector[i] = -1
    l_capped = (-1 / l_vect[i,0]) * l_tilda_vector
    Q = np.identity(n)
    Q[:,i] = l_capped.T
    # for j,v in enumerate(l_capped[0]):
    #     Q[j,i] = v
    return np.asmatrix(Q)


def get_a_dash_inverted(q_mtrx, a_dash_mtrx):
    return q_mtrx * a_dash_mtrx

# n = 3
# i = 1
# A_matrix = np.matrix([
#     [1, 0, 5],
#     [2, 1, 6],
#     [3, 4, 0]
#     ]
# )
# # A_inv_matrix = numpy.linalg.inv(A_matrix)
# A_inv_matrix = np.matrix([
#     [-24, 20, -5],
#     [18, -15, 4],
#     [5, -4, 1]
#     ]
# )

# x_vector = np.matrix([
#     [2],
#     [2],
#     [2]
# ])

def inverse_matrix(A_matrix, A_inv_matrix, x_vector, i):
    n = len(x_vector)
    l_vect = A_inv_matrix * x_vector

    if l_vect[i] == 0:
        #print("l[i] is Zero. Matrix can't be inverted")
        raise ValueError(f"l[{i}] is Zero. Matrix can't be inverted")

    A_dash_matrix = A_matrix.copy()
    A_dash_matrix[:,i] = x_vector

    Q_matrix = get_Q_matrix(l_vect, i, n)
    A_dash_inv = Q_matrix * A_inv_matrix

    return A_dash_inv