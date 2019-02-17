import math
import numpy as np

x_vector__t = [0,0,1,3,2] # 5

A_matrix = np.matrix(
    [[-1, 1, 1, 0, 0],
     [1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1]]
)

c_vect__t = np.matrix([[1,1,0,0,0]])

J_b = [2,3,4] # possibly i-1 ???


def get_A_b_matrix(A_matrix, J_b):
    result = None
    for j in J_b:
        s = A_matrix[:,j:j+1]
        if result is None:
            result = s
        else:
            result = np.append(result, s, axis=1)
    return result


def get_c_b_matrix(c, J_b):
    # c is expected to be of type np.array (flat)
    ar = [c[i] for i in J_b]
    return np.matrix(ar) # returned as transponed already!

def is_optimal_plan_found(estimations_vector, J_b):
    # estimations_vector is expected to be np.array (flat)
    for i,v in enumerate(estimations_vector):
        if i not in J_b and v < 0:
            return i
    return -1

def get_new_basis_index(z_vector, x_vector, J_b):
    # z_, x_ vector are expected to be arrays
    min_q = math.inf
    min_q_index = 0
    for i,z in enumerate(z_vector):
        q = 0 
        if z > 0:
            q = x_vector[J_b[i]]/z
        else:
            q = math.inf
        if min_q > q:
            min_q_index = i
            min_q = q
    return min_q, min_q_index

A_b = get_A_b_matrix(A_matrix, J_b)
A_b_inversed = A_b.I

while True:
    c_b__t = get_c_b_matrix(c_vect__t.A1, J_b)
    u_potentials__t = c_b__t * A_b_inversed
    delta_estimations__t = u_potentials__t * A_matrix - c_vect__t

    first_negative_index  = is_optimal_plan_found(delta_estimations__t.A1, J_b)

    if first_negative_index < 0:
        print("WE'VE GOT A PLAN!")
        print(x_vector__t)
        break
    
    z_vect = A_b_inversed * A_matrix[:,first_negative_index]
    z_vect = z_vect.A1
    q0, q0_index = get_new_basis_index(z_vect, x_vector__t, J_b)
    if q0 == math.inf:
        print("Error. INFINITY function")
        break

    J_b[q0_index] = first_negative_index

    x_new = [0]*len(x_vector__t)
    for i, j in enumerate(J_b):
        x_new[j] = x_vector__t[j] - q0 * z_vect[i]
    x_new[first_negative_index] = q0
    x_vector__t = x_new

    A_b = get_A_b_matrix(A_matrix, J_b)
    A_b_inversed = A_b.I
    




