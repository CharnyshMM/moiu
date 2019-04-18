import numpy as np
from lab5 import build_graph_simple, \
    find_cycle_simple,\
    balance_conditions,\
    new_north_west_plan_builder,\
    get_U_Vs,\
    check_optimality,\
    update_plan
  
# a = [20, 30, 25]
# b = [10, 10, 10, 10, 10]

# c = np.array([
#   [2, 8, -5, 7, 10],
#   [11, 5, 8, -8, -4],
#   [1, 3, 7, 4, 2]
# ])


# a = [20,11,18,27]
# b = [11,4,10,12,8,9,10,4]

# c = np.array([
#   [-3, 6, 7, 12, 6, -3, 2, 16],
#   [4, 3, 7, 10, 0, 1, -3, 7],
#   [19, 3, 2, 7, 3, 7, 8, 15],
#   [1, 4, -7, -3, 9, 13, 17, 22]
# ])


# a = [15,12,18,20]
# b = [5,5,10,4,6,20,10,5]

# c = np.array([
#   [3,10,70,-3,7,4,2,-20],
#   [3,5,8,8,0,1,7,-10],
#   [-15, 1, 0, 0, 13, 5, 4, 5],
#   [1, -5, 9, -3, -4, 7, 16, 25]
# ])

a = [53, 20, 45, 38]
b = [15, 31, 10, 3, 18]

c = np.array([
  [3, 0, 3, 1, 6],
  [2, 4, 10, 5, 7],
  [-2, 5, 3, 2, 9],
  [1, 3, 5, 1, 9]
])



a, b, c = balance_conditions(a, b, c)
print("A:", a)
print("B:", b)
m, n = c.shape

x_plan, J_b = new_north_west_plan_builder(a, b, c)

# print(find_cycle_simple(build_graph_simple(J_b, m,n),m, n))
# J_b.append((0,2))
# J_b.append((0,5))

not_J_b = []
for i in range(m):
  for j in range(n):
    if (i,j) in J_b:
      continue
    not_J_b.append((i,j))

if len(J_b) != m + n - 1:
  raise ValueError(f"this plan needs one more point m+n-1={m+n-1}, len = {len(J_b)}")


iteration = 1
while True:
  print(f"\n#ITERATION {iteration}\n")
  iteration+=1
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
    total = 0
    for i in range(m):
      for j in range(n):
        total += x_plan[i,j] * c[i,j]
    print("Total ", total)
    exit(0)

  J_b.append(max_diff_point)
  graph = build_graph_simple(J_b, m, n)
  cycle = find_cycle_simple(graph,m,n)
  print("cycle:\n", cycle)
  x_plan, J_b = update_plan(x_plan, cycle, J_b, max_diff_point[0], max_diff_point[1])


