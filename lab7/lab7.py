from sympy import Symbol, diff
from scipy.optimize import linprog
from sympy.solvers.inequalities import solve_poly_inequalities
import sys
import numpy as np

from values import B0, B, c, c0, a, x_ast, x_dash, n, m

ROUND_DIGITS = 3

sys.stdout = open("log.txt", "w+")


def get_g_functions(x_symbols, B, c, a):
  g = []
  for i in range(len(B)):
    g_i = 0.5*((x_symbols.T*B[i].T)*B[i])*x_symbols + c[i]*x_symbols + a[i]
    g.append(g_i.sum())
  return g


def get_A_b_constraints(x_symbols, g, x_values):
  I_0 = []
  A_ub = []
  b_ub = []
  for i, g_i in enumerate(g):
    if round(g_i.subs(x_values), ROUND_DIGITS) == 0:
      I_0.append(i)
      b_ub.append(0)
      A_row = []
      for j in range(n):
        dg_dx = diff(g_i, x_symbols[j, 0]).subs(x_values)
        #print("dg_dx_i:", dg_dx)
        A_row.append(dg_dx)
      A_ub.append(A_row)
  return A_ub, b_ub


# def find_a_parameter(df_dx_array, l_0, delta_x):
#   a = Symbol('a')
#   df_dx_vector = np.matrix(df_dx_array)
#   d = (df_dx_vector*l_0 + a*df_dx_vector *delta_x).sum()
#   print(solve_rational_inequalities([
    
#   ]))


x_symbols = np.matrix([[Symbol(f"x{i}") for i in range(n)]]).T
print("x:", x_symbols)

old_x_values = {x_symbols[i, 0]: x_ast[i,0] for i in range(n)}
print("x_values", old_x_values)

# f(x) = 0.5x' B(0) 'B(0)x + c(0)'x,

f = (0.5*((x_symbols.T*B0.T)*B0)*x_symbols + c0*x_symbols).sum()
print("f(x) = ", f, sep='\n')
print("f(old_x) = ", f.subs(old_x_values))


# gi := 0.5x' B(i)' B(i)x + c(i)'x + α(i), i = 1,...,5, x ∈ ||8

g = get_g_functions(x_symbols, B, c, a)
print("g: ", *g, sep='\n')


A_ub, b_ub = get_A_b_constraints(x_symbols, g, old_x_values)

print("---- SUBTASK DATA -----")
print("A_ub: ", A_ub)
print("b_ub: ", b_ub)


objective_function = [diff(f, x_symbols[i, 0]).subs(old_x_values) for i in range(n)]
print("objective function:\n", objective_function)

bounds = []
J_0 = []
for j, x_j in enumerate(x_ast):
  if x_j == 0:
    J_0.append(j)
    bounds.append((0, 1))
  else:
    bounds.append((-1, 1))
print("bounds ", bounds)

result = linprog(objective_function, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
l_0 = np.matrix(result.x)
subtask_function_val = result.fun
print("l_0: ", l_0)
print("df_dx(old_x)*l_0 = ",subtask_function_val)

if (subtask_function_val >= 0):
  print("x_0 is already a great plan")
  exit(0)


t = 0.5
a = 1

delta_x = x_dash - x_ast

better_x = x_ast + t*(l_0.T + a*delta_x)

print("------ PLAN IMPROVED -----")
print("better_x: ", better_x)

new_x_values = {x_symbols[i,0]: better_x[i, 0] for i in range(n)}

print("f(old_x) = ", f.subs(old_x_values))
print("f(better_x):= ", f.subs(new_x_values))
