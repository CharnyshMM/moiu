from functions import get_A_b_matrix, get_c_b
import numpy as np
import math
import sys

from ordered_set import OrderedSet

sys.stdout = open("log.txt", "w+")

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
d = np.matrix([[7, 3, 3]]).T

D_matrix = B_matrix.T * B_matrix

c = (-d.T * B_matrix).T

x = np.matrix([[0,0, 6, 4, 5, 0, 0, 0]]).T
J = {i for i in range(n)}
J_op = OrderedSet([2,3,4])
J_ast = OrderedSet([2,3,4])

def get_c_x(x, D_matrix):
  return D_matrix * x + c

def get_H_matrix(D_martix, A_matrix, J_ast):
  D_ast_matrix = np.zeros((len(J_ast), len(J_ast)))
  for i_index, i in enumerate(J_ast):
    for j_index, j in enumerate(J_ast):
      D_ast_matrix[i_index, j_index] = D_martix[i,j]

  A_ast_matrix = get_A_b_matrix(A_matrix, J_ast)
  H_matrix = np.append(D_ast_matrix, A_ast_matrix, axis=0)
  A_ast_matrix_inv = A_ast_matrix.I

  m_of_the_half, _ = H_matrix.shape
  m_of_the_A_ast_I, n_of_the_A_ast_I = A_ast_matrix_inv.shape
  m_of_zeros = m_of_the_half - m_of_the_A_ast_I
  n_of_zeros = n_of_the_A_ast_I
  H_matrix_right_half = np.append(A_ast_matrix_inv, np.zeros((m_of_zeros, n_of_zeros)), axis=0)
  H_matrix = np.append(H_matrix, H_matrix_right_half, axis=1)
  return H_matrix


def get_bb_vector(D_matrix, A_matrix, j_0, J_ast):
  bb = [D_matrix[j,j_0] for j in J_ast] 
  bb.extend(A_matrix[:,j_0].A1)
  return np.matrix([bb]).T
    

def get_directions_vector(D_matrix, A_matrix, j_0, J, J_ast):
  l = [0]*len(J)
  H_matrix = get_H_matrix(D_matrix, A_matrix, J_ast)
  print("H:")
  print(H_matrix)
  bb = get_bb_vector(D_matrix, A_matrix, j_0, J_ast)
  print("bb:")
  print(bb)

  bl = (-H_matrix.I * bb).A1[:len(J_ast)]
  jj = 0
  for j in J_ast:
    l[j]=bl[jj]
    jj+=1
  l[j_0] = 1

  print("l:")
  print(l)
  return np.matrix([l]).T

def get_min_tetta_step(l, x, J_ast):
  min_tetta = math.inf
  min_tetta_index = -1
  for j in J_ast:
    l_j = l[j]
    tetta = math.inf
    if l_j < 0:
      tetta = -x[j] / l_j
    if tetta < min_tetta:
      min_tetta = tetta
      min_tetta_index = j
  print("counted min tetta:")
  print(min_tetta,"index = (", min_tetta_index, ")")
  return min_tetta_index, min_tetta


iteration = 1
# iteration 1
while (True):
  print("\n\n iteration: ", iteration)

  print("J_ast")
  print(J_ast)

  print("J_op")
  print(J_op)
  
  print("step 1:")
## step 1
  A_op = get_A_b_matrix(A_matrix, J_op)
  print("A_op:")
  print(A_op)
  c_dash = D_matrix*x + c
  c_dash_op = get_A_b_matrix(c_dash.T, J_op).T
  print("c_dash_op")
  print(c_dash_op)

  ## step 2
  u = -c_dash_op.T * A_op.I # should be T???
  delta = u*A_matrix
  delta = delta.T + c_dash
  
  print("delta:")
  print(delta)
  # сделать, чтобы только те индексы, что не в J_Ast
  # checking optimality
  j_0 = -1
  j_0_value = None
  for i, d in enumerate(delta.A1):
    if i in J_ast:
      continue
    if d < 0:
      j_0 = i
      j_0_value = d
      break

  if j_0_value is None:
    print("OPTIMAL PLAN FOUND!!!")
    print(x)
    result = c.T * x + 0.5 * x.T * D_matrix * x
    print("result:= ", result)
    exit(0)

  print("j_0:= ", j_0)
  ## step 3
  while True:
    iteration += 1
    print("step 3:")
    l = get_directions_vector(D_matrix, A_matrix, j_0, J, J_ast)

    ## step 4
    little_delta = l.T*D_matrix*l
    tetta_j0 = math.inf
    if little_delta > 0: # >???????
      tetta_j0 = abs(delta[j_0]) / little_delta

    j, tetta_j_ast = get_min_tetta_step(l.A1, x.A1, J_ast)

    tetta_final = tetta_j_ast
    j_ast = j
    if tetta_j0 < tetta_j_ast:
      tetta_final = tetta_j0
      j_ast = j_0

    if (tetta_final == math.inf):
      print("FUNCTION iS NOT LIMITED")
      exit(1)
    print("tetta final:")
    print(tetta_final)
    ## step 5
    x = x + tetta_final.sum()*l

    print("x")
    print(x)
    ## step 6
    # updating J_ast & J_op

    if j_0 == j_ast:
      print("s1")
      J_ast.add(j_0)
      print("J_Ast update:")
      print(J_ast)
      break # goto 1

    if j_ast in J_ast.difference(J_op):
      print("s2")
      J_ast.remove(j_ast)
      delta[j_0] = delta[j_0] + tetta_final*little_delta

      print("J_Ast update:")
      print(J_ast)

      print("delta update:")
      print(delta)
      # goto 3
      continue

    if j_ast in J_op and len(J_op.symmetric_difference(J_ast)) == 0:
      print("s4_2")
      J_op.remove(j_ast)
      J_op.append(j_0)
      J_ast.remove(j_ast)
      J_ast.append(j_0)

      print("J_Ast update:")
      print(J_ast)

      print("J_op update:")
      print(J_op)
      break # goto 1

    if j_ast in J_op:
      # 6c
      print("s3")
      s = J_op.index(j_ast)
      for j in J_ast.difference(J_op):
        if (A_op.I*A_matrix[:,j])[s] != 0:
          J_op.remove(j_ast)
          J_op.append(j)
          J_ast.remove(j_ast)
          delta[j_0] = delta[j_0] + tetta_final*little_delta

          print("J_Ast update:")
          print(J_ast)

          print("J_op update:")
          print(J_op)

          print("delta update:")
          print(delta)

          print("goto 3")
          break
      else:
        print("s4_1")
        J_op.remove(j_ast)
        J_op.append(j_0)
        J_ast.remove(j_ast)
        J_ast.append(j_0)

        print("J_Ast update:")
        print(J_ast)

        print("J_op update:")
        print(J_op)
        print("goto1")
        break # goto 1
