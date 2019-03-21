import math
import numpy as np
from main_simplex import solve, get_A_b_matrix


# b = np.matrix([[0],[0]])

# A = np.matrix([[1,1,1],[2,2,2]])

# x = np.matrix([[0,0,0]])


# Ex. 2
A = np.matrix([
[0, 1, 4, 1, 0, -3, 5, 0],
[1, -1, 0, 1, 0, 0, 1, 0],
[0, 7, -1, 0, -1, 3, 8, 0],
[1, 1, 1, 1, 0, 3, -3, 1]
], float)
c_vector = np.matrix([[-5, -2, 3, -4, -6, 0, -1, -5]])
b = np.matrix([[6], [10], [-2], [15]])
# answer = [10.0, 0, 2.2, 0, 2.6, 0.9333333333333333, 0, 0]


# # Ex. №5
# A = np.matrix([
#     [1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3]
# ], dtype=float)
# b = np.matrix([[0], [0], [0]])
# c_w = [0, 1, 1]
# # answer = [0, 0, 0]

# # Ex. №4
# A = np.matrix([
# [0, 1, 1, 1, 0, -8, 1, 5,],
# [0, -1, 0, -7.5, 0, 0, 0, 2],
# [0, 2, 1, 0, -1, 3, -1.4, 0],
# [1, 1, 1, 1, 0, 3, 1, 1]
# ], dtype=float)
# b = np.matrix([[15], [-45], [1.8], [19]])
# c_vector = np.matrix([[-6, -9, -5, 2, -6, 0, 1, 3]])
# # answer = [0, 0, 0, 7.055478884902015, 0, 1.802925752139111, 2.577698040298096, 3.9580458183825558]

# # Ex. №3
# A = np.matrix([
# [0, 1, 4, 1, 0, -8, 1, 5],
# [0, -1, 0, -1, 0, 0, 0, 0],
# [0, 2, -1, 0, -1, 3, -1, 0],
# [1, 1, 1, 1, 0, 3, 1, 1]
# ])
# c_vector = np.matrix([[-5, 2, 3, -4, -6, 0, 1, -5]])
# b = np.matrix([[36], [-11], [10], [20]])
# # answer = [0, 9.5, 5.333333333333333, 1.5, 0, 0, 3.6666666666666665, 0]


# A = np.matrix([
# [2, 1,  3],
# [1, -3, 1],
# [1, 11, 3]
# ])
# c_w = [-5, 2, 3, -4, -6, 0, 1, -5]
# b = np.matrix([[1], [-3], [11]])


def foreach_predicate(iterable, predicate):
    for i, item in enumerate(iterable):
        if not predicate(item):
            return i, item
    return -1, None


def prepare_A_b_matrixes(A_matrix, b_vector):
    A_matrix_copy = A_matrix.copy()
    b_vector_copy = b_vector.copy()

    for i, b_i in enumerate(b_vector_copy.A1):
        if b_i >= 0:
            continue
        
        for j, a_i_j in enumerate(A_matrix_copy[i].A1):
            A_matrix_copy[i,j] = - a_i_j
        b_vector_copy[i,0] = -b_i

    return A_matrix_copy, b_vector_copy


A_matrix, b_vector = prepare_A_b_matrixes(A, b)

m, n = A_matrix.shape
eye = np.eye(m, k=0)
extended_A_matrix = np.append(A_matrix, eye, axis=1)


x_extended = [0]*(n) + [i for i in b_vector.A1]
J_b_for_x_extended = [i for i in range(n, n+m)]
c_for_extended_t = np.matrix([[0]*(n) + [-1]*(m)])

x_extended, J_b_for_x_extended = solve(extended_A_matrix, c_for_extended_t, x_extended, J_b_for_x_extended)

for x_artificial in x_extended[n:]:
    if x_artificial != 0:
        raise ValueError("The problem is not solvable")

X_basis_plan = x_extended[:n] # ???? n-1
J_b = J_b_for_x_extended

k, j_k = foreach_predicate(J_b, lambda j_i: j_i < n)

while not k < 0:
    A_ast_b = get_A_b_matrix(extended_A_matrix, J_b)
    
    i_0 = 0
    e_k = np.zeros(m)
    e_k[k] = 1
    for j in range(m):
        if j in J_b:
            continue
        
        A_j = A_matrix[:,j]
        alpha_j = e_k * A_ast_b.I * A_j

        j_0, alpha_j0 = foreach_predicate(alpha_j.A1, lambda j_i: j_i == 0) #????
        if j_0 != -1:
            del J_b[k]
            J_b.append(j_0)
            break
    else:
        # limitation with index i_0 in problem is linear-dependent
        del J_b[k]
        A_matrix = np.delete(A_matrix, i_0, 0)
        A = np.delete(A, i_0, 0)
        extended_A_matrix = np.delete(extended_A_matrix, i_0, 0)
        b_vector = np.delete(b_vector, i_0, 0)
        break

    k, j_k = foreach_predicate(J_b, lambda j_i: j_i < n)

print("SUCCESS!:")
print("A:\n", A_matrix)
print("b:\n", b_vector)
print("J_b: ", J_b)
print("X_basis_plan: ", X_basis_plan)

x_final, J_b_final = solve(A_matrix, c_vector, X_basis_plan, J_b)
print("X: ", x_final)


exit(0)


