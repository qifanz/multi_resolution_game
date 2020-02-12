from scipy.optimize import linprog
import numpy as np


def __linprog_solver_col(value_matrix):
    m, n = value_matrix.shape

    # solve col
    # objectif vector c is 0*x1+0*x2+...+0*xn+v
    C = []
    for i in range(n):
        C.append(0)
    C.append(-1)
    A = []
    for i_col in range(n):
        col = value_matrix[:, i_col]
        constraint_row = []
        for item in col:
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
    return res['x'][:-1], -res['fun']


def __linprog_solver_row(value_matrix):
    policy, value = __linprog_solver_col(-value_matrix.T)
    return policy, -value


def linprog_solve(value_matrix):
    # rps = nash.Game(np.array(value_matrix))
    # eqs = rps.support_enumeration()
    px, value = __linprog_solver_row(value_matrix)
    py, v2 = __linprog_solver_col(value_matrix)
    # policy_x, policy_y = list(eqs)[0]
    return value, py, px


def run():
    v = [
        [4, 3, 7],
        [1, 2, 3],
        [2, 4, 6]
    ]
    v = np.array(v)
    policy_x, value_x = __linprog_solver_row(v)
    policy_y, value_y = __linprog_solver_col(v)
    linv, linx, liny = linprog_solve(v)
    print('done')
