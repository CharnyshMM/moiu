import numpy as np

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
    c_copy = np.append(c, np.zeros((1, m)), axis=0)
    return a, b, c_copy
  
  if sum_b > sum_a:
    a.append(sum_b - sum_a)
    c_copy = np.append(c, np.zeros((n, 1)), axis=1)
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
    diff = _b[i] - _a[j]
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


def get_U_Vs(c, J_b):
  u = [0]
  v = []

  for i,j in J_b:
    if len(u) > i:
      v.append(c[i,j] - u[i])
    elif len(v) > j:
      u.append(c[i,j] - v[i])

  return u, v


def check_optimality(u, v, c, J_b):
  for i, u_i in enumerate(u):
    for j, v_j in enumerate(v):
      if (i,j) in J_b:
        continue
      
      if u_i + v_j > c[i,j]:
        return False, (i, j)

  return True, None


def build_graph(J_b):
    graph = []

    for k, (i_0, j_0) in enumerate(J_b):
        for i, j in J_b[k:]:
            if i == i_0 and j == j_0:
                continue
            if i == i_0:
                graph.append(((i_0, j_0), (i, j)))
                graph.append(((i, j), (i_0, j_0)))
                break
        for i, j in J_b[k:]:
            if i == i_0 and j == j_0:
                continue
            if j == j_0:
                graph.append(((i_0, j_0), (i, j)))
                graph.append(((i, j), (i_0, j_0)))
                break
    return graph

def build_graph_simple(J_b, m, n):
    graph = np.zeros((m,n))
    
    for i, j in J_b:
        graph[i,j] = 1
    return graph


def find_cycle_simple(graph, m, n):
    something_deleted = True
    deletings_count = 0
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

  
def update_plan(x, cycle_in, i_0, j_0):
    visited_mark = 2
    m,n = x.shape
    q = x.min()
    cycle = cycle_in.copy()
    
    i = i_0
    j = j_0
    next_i = i_0
    next_j = j_0

    while True:
        cycle[i,j] = visited_mark
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
    
    cycle = cycle * 0.5 * q
    return cycle
    
    
  
 