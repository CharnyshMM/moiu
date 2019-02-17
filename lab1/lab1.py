import numpy as np


def get_l_vector(a_inv_mtrx, x_vect):
    return a_inv_mtrx * x_vect

def get_Q_matrix(l_vect, i, n):
    l_tilda_vector = l_vect.copy()
    l_tilda_vector[i] = -1
    l_capped = (-1 / l_vect[i,0]) * l_tilda_vector
    Q = np.identity(n)
    for j,v in enumerate(l_capped[0]):
        Q[j,i] = v
    return np.asmatrix(Q)


def get_a_dash_inverted(q_mtrx, a_dash_mtrx):
    return q_mtrx * a_dash_mtrx

n = 3
i = 1
A_matrix = np.matrix([
    [1, 0, 5],
    [2, 1, 6],
    [3, 4, 0]
    ]
)
# A_inv_matrix = numpy.linalg.inv(A_matrix)
A_inv_matrix = np.matrix([
    [-24, 20, -5],
    [18, -15, 4],
    [5, -4, 1]
    ]
)

x_vector = np.matrix([
    [2],
    [2],
    [2]
])


l_vect = get_l_vector(A_inv_matrix, x_vector)
print("l:= ", l_vect)

if l_vect[i] == 0:
    print("l[i] is Zero. Matrix can't be inverted")
    exit(1)

A_dash_matrix = A_matrix.copy()

for j, v in enumerate(x_vector):
    print(v)
    A_dash_matrix[j,i] = v[0][0]

print("A dash:")
print(A_dash_matrix)

Q_matrix = get_Q_matrix(l_vect, i, n)
print("Q:")
print(Q_matrix)
A_dash_inv = get_a_dash_inverted(Q_matrix, A_dash_matrix)

print("A dash inv: ", A_dash_inv)