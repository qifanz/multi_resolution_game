from scipy.optimize import linprog
import numpy as np


def linprog_solver(value_matrix):
    m, n = value_matrix.shape

    # solve col
    # objectif vector c is 0*x1+0*x2+...+0*xn+v
    C = []
    for i in range(n):
        C.append(0)
    C.append(-1)
    A = []
    for row in value_matrix:
        constraint_row = []
        for item in row:
            constraint_row.append(-item)
        constraint_row.append(1)
        A.append(constraint_row)
    B = []
    for i in range(m):
        B.append(0)

    A_eq = []
    A_eq_row = []
    for i in range(n):
        A_eq_row.append(1)
    A_eq_row.append(0)
    A_eq.append(A_eq_row)
    B_eq = [1]

    bounds = []
    for i in range(n):
        bounds.append((0, 1))
    bounds.append((None, None))

    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq, bounds=bounds)
    return res['x'][:,-1], res['fun']


def run():
    value_matrix = [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0]
    ]
    res = linprog_solver(np.array(value_matrix))
    print('done')


run()
