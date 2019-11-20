from nash_solver import linprog_solver
import numpy as np


class MatrixGameSolver:
    def __init__(self, game):
        self.game = game
        self.alpha = 0.5  # convergence factor
        self.mu = 0.6
        self.gamma = 0.9
        self.value_vector = np.zeros(game.get_n_states())  # set 0 for initial estimation

    def solve(self):
        pass

    def create_action_value_matrix(self, state):
        reward_matrix = self.game.get_action_reward(state)
        transition_matrix = self.game.get_action_transition(state)
        action_value_matrix = reward_matrix.copy()

        n_p1_actions, n_p2_actions = reward_matrix.shape

        for i in range(n_p1_actions):
            for j in range(n_p2_actions):
                prob, next_state = transition_matrix[i, j]
                action_value_matrix[i, j] += self.gamma * prob * self.value_vector[next_state]

    def solve_state(self, state):
        action_value_matrix = self.create_action_value_matrix(state)
        linprog_res = linprog_solver(action_value_matrix)
        return linprog_res

    def calc_L(self):
        L_v = []
        for state in range(self.game.get_n_states()):
            L_v.append(self.solve_state(state))
        return L_v

    def calc_psi(self, L_v):
        return np.subtract(L_v, self.value_vector)

    def calc_J(self, psi_v):
        return 0.5 * np.dot(psi_v.T, psi_v)
