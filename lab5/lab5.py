import numpy as np
import sympy as sp
import math

a = []
b = []

c = np.array([[]])

def balance_conditions(a, b, c):
  sum_a = sum(a)
  sum_b = sum(b)
  if sum_a == sum_b:
    return a, b, c
  
  m,n = c.shape
  if sum_a > sum_b:
    b.append(sum_a - sum_b)
    c_copy = np.append(c, np.zeros((m,1)), axis=1)
    return a, b, c_copy
  
  if sum_b > sum_a:
    a.append(sum_b - sum_a)
    c_copy = np.append(c, np.zeros((1, n)), axis=0)
    return a, b, c_copy

def north_west_basis_plan(a, b, c):
  _a = a.copy()
  _b = b.copy()
  m,n = c.shape
  X_plan = np.zeros((m,n))
  J_b = []

  i = 0
  j = 0
  while i < m or j < n :
    if _a[i] == _b[j] == 0:
      if i < m-1:
        i+=1
      else:
        j+=1
    diff = _b[j] - _a[i]
    X_plan[i, j] = abs(diff)
    J_b.append((i, j))
    if diff >= 0:
      _a[i] = 0
      _b[j] = diff
      i+=1
    elif diff < 0:
      _a[i] = -diff
      _b[j] = 0
      j+=1
  
  # checking degenerateness ???
  return X_plan, J_b


def new_north_west_plan_builder(a__in,b__in, c):
  a = a__in.copy()
  b = b__in.copy()
  m,n = c.shape
  X_plan = np.zeros((m,n))
  J_b = []
  j = 0
  for i in range(m):
    while a[i] != 0:
      diff = b[j] - a[i]
      J_b.append((i,j))
      if (diff > 0):
        X_plan[i,j] = a[i]
        a[i] = 0
        b[j] = diff
      else:
        X_plan[i,j] = b[j]
        a[i] = -diff
        b[j] = 0
        j+=1

  return X_plan, J_b
      

def get_U_Vs(c, J_b):
  system = []
  m,n = c.shape
  u = [sp.Symbol(f"u{i}") for i in range(m)]
  v = [sp.Symbol(f"v{j}") for j in range(n)]

  system.append(u[0])
  for i,j in J_b:
    system.append(u[i] + v[j]- c[i, j])
  
  results = sp.solve(system, u+v)
  results = results
  results = list(results.values())
  u = results[:m]
  v = results[m:]
  return u, v

def check_optimality(u, v, c, J_b):
  max_one = -math.inf
  max_one_point = (-1, -1)
  for i, u_i in enumerate(u):
    for j, v_j in enumerate(v):
      if (i,j) in J_b:
        continue
      
      if u_i + v_j > c[i,j]:
        if max_one < u_i + v_j:
          max_one = u_i + v_j
          max_one_point = (i, j)

  if max_one_point[0] > 0:
    return False, max_one_point
  return True, None


def build_graph_simple(J_b, m, n):
    graph = np.zeros((m,n))
    
    for i, j in J_b:
        graph[i,j] = 1
    return graph


def find_cycle_simple(graph_in, m, n):
    something_deleted = True
    graph = graph_in.copy()
    while something_deleted:
        something_deleted = False
        for i,row in enumerate(graph):
            if sum(row) != 1:
                continue
            graph[i] = np.zeros(n)
            something_deleted = True
        for j in range(n):
            if sum(graph[:, j]) != 1:
                continue
            graph[:, j] = np.zeros(m)
            something_deleted = True
    return graph
  
def get_min_element_of_matrix(mtx, points):
  m,n = mtx.shape
  min_point = (None, None)
  min_value = math.inf
  for i,j in points:
      if min_value > mtx[i,j]:
        min_value = mtx[i,j]
        min_point = (i,j)
  return min_value, min_point

def update_plan(x__in, cycle__in, J_b__in, i_0, j_0):
    visited_mark = 2
    m,n = x__in.shape
   
    cycle = cycle__in.copy()
    
    i = i_0
    j = j_0
    next_i = i_0
    next_j = j_0

    U_minus = []
    while True:
        cycle[i,j] = visited_mark
        if visited_mark < 0:
          U_minus.append((i,j))
        visited_mark = -visited_mark
        
        for jj in range(0, n):
            if cycle[i, jj] == 1:
                next_j = jj
                break
        else:       
            for ii in range(0, m):
                if cycle[ii, j] == 1:
                    next_i = ii
                    break
        if i == next_i and j == next_j:
            break
        i = next_i
        j = next_j
    
    q, (q_i, q_j) = get_min_element_of_matrix(x__in, U_minus)
    cycle = cycle * 0.5 * q
    x = x__in + cycle
    J_b = J_b__in.copy()
    J_b.remove((q_i,q_j))

    return x, J_b
  
 