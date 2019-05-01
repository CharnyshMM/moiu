from functions import get_A_b_matrix, get_c_b, foreach_predicate
import numpy as np
import math
from ordered_set import OrderedSet

A_matrix = np.matrix([
  [1, 2, 0, 1, 0, 4, -1, -3],
  [1, 3, 0, 0, 1, -1, -1, 2],
  [1, 4, 1, 0, 0, 2, -2, 0]
])
b = np.matrix([[4,5,6]]).T
m,n = A_matrix.shape

B_matrix = np.matrix([
  [1, 1, -1, 0, 3, 4, -2, 1],
  [2, 6, 0, 0, 1, -5, 0, -1],
  [-1, 2, 0, 0, -1, 1, 1, 1]
])
d = np.matrix([[7, 3, 3]])

D_matrix = B_matrix.T * B_matrix

c = (-d.T * B_matrix).T

x = np.matrix([[0,0, 6, 4, 5, 0, 0, 0]]).T
J = {i for i in range(m)}
J_op = OrderedSet([3,4,5])
J_ast = OrderedSet([3,4,5])

def get_c_x(x, D_matrix):
  return D_matrix * x + c

def get_H_matrix(D_martix, A_matrix, J_ast):
  D_ast_matrix = np.zeros((len(J_ast),len(J_ast)))
  for i in J_ast:
    for j in J_ast:
      D_ast_matrix[i, j] = D_martix[i,j]

  A_ast_matrix = get_A_b_matrix(A_matrix, J_ast)
  H_matrix = np.append(D_ast_matrix, A_ast_matrix, axis=0)
  A_ast_matrix_inv = A_ast_matrix.I
  H_matrix_right_half = np.append(A_ast_matrix_inv, np.zeros(A_ast_matrix_inv.shape), axis=0)
  H_matrix = np.append(H_matrix, H_matrix_right_half, axis=1)
  return H_matrix


def get_bb_vector(D_matrix, A_matrix, j_0, J_ast):
  bb = [D_matrix[j,j_0] for j in J_ast] + A_matrix[:,j_0].A1
  return np.matrix([bb]).T
    

def get_directions_vector(D_matrix, A_matrix, j_0, J, J_ast):
  l = [0]*len(J.difference(J_ast))
  H_matrix = get_H_matrix(D_matrix, A_matrix, J_ast)
  bb = get_bb_vector(D_matrix, A_matrix, j_0, J_ast)
   
  l += (-H_matrix.I * bb).A1
  l[j_0] = 1

  return np.matrix([l]).T

def get_min_tetta_step(l, x, J_ast):
  min_tetta = math.inf
  min_tetta_index = -1
  for j, l_j in enumerate(l.A1):
    tetta = math.inf
    if l_j < 0:
      tetta = -x[j] / l_j
    if tetta < min_tetta:
      min_tetta = l_j
      min_tetta_index = j
  return min_tetta_index, min_tetta



# iteration 1
while (True):
## step 1
  A_op = get_A_b_matrix(A_matrix, J_op)
  c_dash = get_c_x(x, D_matrix)
  c_dash_op = get_c_b(c_dash, J_op)


  ## step 2
  u = -c_dash_op.T * A_op.I # should be T???
  delta = u*A_matrix + c_dash

  # checking optimality
  j_0, j_0_value = foreach_predicate(delta, lambda x: x >= 0)

  if j_0 is None:
    print("OPTIMAL PLAN FOUND!!!")
    print(x)
    exit(0)

  
  ## step 3
  while True:
    l = get_directions_vector(D_matrix, A_matrix, j_0, J, J_ast)

    ## step 4
    little_delta = l.T*D_matrix*l
    tetta_j0 = math.inf
    if little_delta > 0: # >???????
      tetta_j0 = abs(delta[j_0]) / little_delta

    j, tetta_j_ast = get_min_tetta_step(l, x.A1, J_ast)

    tetta_final = tetta_j_ast
    j_ast = j
    if tetta_j0 < tetta_j_ast:
      tetta_final = tetta_j0
      j_ast = j_0

    if (tetta_final == math.inf):
      print("FUNCTION iS NOT LIMITED")
      exit(1)

    ## step 5
    x = x + tetta_final*l

    ## step 6
    # updating J_ast & J_op

    if j_0 == j_ast:
      J_ast.add(j_0)
      print("goto 1")

    if j_ast in J_ast.difference(J_op):
      # goto 3
      J_ast = J_ast.remove(j_ast)
      delta[j_0] = delta[j_0] + tetta_final*little_delta

    if j_ast in J_op and len(J_op.difference(J_ast)) == 0:
      J_op.remove(j_ast)
      J_op.append(j_0)
      J_ast.remove(j_ast)
      J_ast.append(j_0)
      break # goto 1

    if j_ast in J_op:
      # 6c
      s = J_op.index(j_ast)
      for j in J_ast.difference(J_op):
        if (A_op.I*A_matrix[:,j])[s] != 0:
          J_op.remove(j_ast)
          J_op.append(j)
          J_ast.remove(j_ast)
          delta[j_0] = delta[j_0] + tetta_final*little_delta
          print("goto 3")
          break
      else:
        J_op.remove(j_ast)
        J_op.append(j_0)
        J_ast.remove(j_ast)
        J_ast.append(j_0)
        break # goto 1
