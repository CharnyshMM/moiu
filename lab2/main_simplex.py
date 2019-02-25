import math
import numpy as np
from lab1 import inverse_matrix

# x_vector__t = [0,0,1,3,2] # 5

# A_matrix = np.matrix(
#     [[-1, 1, 1, 0, 0],
#      [1, 0, 0, 1, 0],
#      [0, 1, 0, 0, 1]]
# )

# c_vect__t = np.matrix([[1,1,0,0,0]])

# J_b = [2,3,4]

c_vect__t = np.matrix([[-5, -2, 3, -4, -6, 0, -1, -5]])

x_vector__t = [4, 0, 0, 6, 2, 0, 0, 5]

A_matrix = np.matrix(
    [
    [0,1,4,1,0,-3,5,0],
    [1,-1,0,1,0,0,1,0],
    [0,7,-1,0,-1,3,8,0],
    [1,1,1,1,0,3,-3,1]
    ]
)



J_b = [0,3,4,7]

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
print("A_b = \n", A_b)
A_b_inversed = A_b.I
iterations_count = 1

while True:
    print(f"\n# {iterations_count} \n")
    iterations_count += 1

    c_b__t = get_c_b_matrix(c_vect__t.A1, J_b)
    print(f"C_b = {c_b__t}")
    u_potentials__t = c_b__t * A_b_inversed
    print(f"U_potentials = {u_potentials__t}")
    delta_estimations__t = u_potentials__t * A_matrix - c_vect__t
    print(f"Delta_estimations = {delta_estimations__t}")

    first_negative_index  = is_optimal_plan_found(delta_estimations__t.A1, J_b)

    if first_negative_index < 0:
        print("\n************************")
        print("WE'VE GOT A PLAN!")
        print(x_vector__t)

        print("c'*x0 = ", c_vect__t*np.matrix([x_vector__t]).T)
        break
    
    z_vect = A_b_inversed * A_matrix[:,first_negative_index]
    z_vect = z_vect.A1
    print(f"Z = {z_vect}")
    q0, q0_index = get_new_basis_index(z_vect, x_vector__t, J_b)
    if q0 == math.inf:
        print("\nError. INFINITY function")
        break

    print(f"Tetta_{q0_index} = {q0}")
    J_b[q0_index] = first_negative_index
    print(f"New J_b = {J_b}")

    x_new = [0]*len(x_vector__t)
    for i, j in enumerate(J_b):
        x_new[j] = x_vector__t[j] - q0 * z_vect[i]
    x_new[first_negative_index] = q0
    x_vector__t = x_new
    print(f"X = {x_vector__t}")

    new_column = A_matrix[:,first_negative_index:first_negative_index+1]

    A_b_inversed = inverse_matrix(
        A_b, 
        A_b_inversed, 
        new_column,
        q0_index
    )

    A_b[:,q0_index] = new_column
    #A_b_inversed = A_b.I

    print("A_b =\n",A_b)
    print("A_b_inversed =\n", A_b_inversed)
    




