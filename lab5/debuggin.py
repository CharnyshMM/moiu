import numpy as np
from lab5 import build_graph_simple, \
    find_cycle_simple,\
    balance_conditions,\
    new_north_west_plan_builder,\
    get_U_Vs,\
    check_optimality,\
    update_plan
  
a = [20, 30, 25]
b = [10, 10, 10, 10, 10]

c = np.array([
  [2, 8, -5, 7, 10],
  [11, 5, 8, -8, -4],
  [1, 3, 7, 4, 2]
])



a, b, c = balance_conditions(a, b, c)
m, n = c.shape

x_plan, J_b = new_north_west_plan_builder(a, b, c)
J_b.append((0,2))
J_b.append((0,5))

not_J_b = []
for i in range(m):
  for j in range(n):
    if (i,j) in J_b:
      continue
    not_J_b.append((i,j))

if len(J_b) < m + n - 1:
  raise ValueError("this plan needs one more point")



while True:
  print("\n#ITERATION:\n")

  print("starting X:\n", x_plan)
  print("starting J_b: ", J_b)
  u, v = get_U_Vs(c, J_b)
  print("u = ",u)
  print("v = ",v)
  is_optimal, max_diff_point = check_optimality(u, v, c, J_b)
  print("max_diff_point = ", max_diff_point)

  if (is_optimal):
    print("We've got a plan!")
    print(x_plan)
    exit(0)

  J_b.append(max_diff_point)
  graph = build_graph_simple(J_b, m, n)
  cycle = find_cycle_simple(graph,m,n)
  print("cycle:\n", cycle)
  x_plan, J_b = update_plan(x_plan, cycle, J_b, max_diff_point[0], max_diff_point[1])


