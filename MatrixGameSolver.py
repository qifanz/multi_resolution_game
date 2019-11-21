from nash_solver import linprog_solver
import numpy as np
from Actions import *


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
        transition_matrix = self.game.get_state_transition(state)

        immediate_reward = self.game.get_state_reward(state)
        action_value_matrix = np.ones((N_ACTIONS, N_ACTIONS)) * immediate_reward

        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                transition_vector = transition_matrix[i, j][state]
                for next_state in range(len(transition_vector)):
                    probability = transition_vector[next_state]
                    action_value_matrix[i, j] += self.gamma * probability * self.value_vector[next_state]
        return action_value_matrix

    def solve_state(self, state):
        action_value_matrix = self.create_action_value_matrix(state)
        policy_x, value = linprog_solver(action_value_matrix)
        return policy_x, value

    def calc_L(self):
        L_v = []
        for state in range(self.game.get_n_states()):
            _, value = self.solve_state(state)
            L_v.append(value)
        return L_v

    def calc_psi(self, L_v):
        return np.subtract(L_v, self.value_vector)

    def calc_J(self, psi_v):
        return 0.5 * np.dot(psi_v.T, psi_v)

    def cal_D(self, state):
        pass
