from nash_solver import *
import numpy as np
from Actions import *


class MatrixGameSolver:
    def __init__(self, game):
        self.game = game
        self.alpha = 0.5  # convergence factor
        self.mu = 0.6
        self.gamma = 0.9
        self.beta = 0.8
        self.value_vector = np.zeros(game.get_n_states())  # set 0 for initial estimation

    def solve(self):
        converge_flag = False
        while not converge_flag:
            L_v, policy = self.calc_L()
            psi_v = self.calc_psi(L_v, self.value_vector)
            J_v = self.calc_J(psi_v)
            if J_v == 0:
                converge_flag = True
            else:
                D_k, I_subtract_P = self.cal_D()
                w = 1
                while True:
                    if self.test_inequality(D_k, w, J_v, psi_v, I_subtract_P):
                        self.value_vector = np.add(self.value_vector, w*D_k)
                    else:
                        w = self.mu * w
        return self.value_vector, policy


    def create_action_value_matrix(self, state, L_v, use_Lv):
        if use_Lv:
            value_vector = L_v
        else:
            value_vector = self.value_vector
        immediate_reward = self.game.get_state_reward(state)
        action_value_matrix = np.ones((N_ACTIONS, N_ACTIONS)) * immediate_reward

        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                transition_vector = self.game.get_state_transition((i, j))[state]
                for next_state in range(len(transition_vector)):
                    probability = transition_vector[next_state]
                    action_value_matrix[i, j] += self.gamma * probability * value_vector[next_state]
        return action_value_matrix

    def solve_state(self, state, L_v=None, use_Lv=False):
        action_value_matrix = self.create_action_value_matrix(state, L_v, use_Lv)
        value, policy_x, policy_y = linprog_solve(np.array(action_value_matrix))
        return policy_x, policy_y, value

    def calc_L(self, new_v=None, use_new_v=False):
        L_v = []
        policy = []
        for state in range(self.game.get_n_states()):
            policy_x, policy_y, value = self.solve_state(state, new_v, use_new_v)
            L_v.append(value)
            policy.append((policy_x, policy_y))
        return L_v, policy

    def calc_psi(self, L_v, v):
        return np.subtract(L_v, v)

    def calc_J(self, psi_v):
        return 0.5 * np.dot(psi_v.T, psi_v)

    def cal_D(self, policy, psi_v):
        n_states = self.game.get_n_states()
        I = np.identity(n_states)
        for state in range(self.game.get_n_states()):
            policy_x = policy[state][0]
            policy_y = policy[state][1]
            for x in range(N_ACTIONS):
                for y in range(N_ACTIONS):
                    I[state] = np.subtract(I[state], self.beta * policy_x[x] * policy_y[y] *
                                           self.game.get_state_transition((x, y))[state])

        return np.linalg.inv(I) * psi_v, I

    def calc_delta_J(self, psi_v, I_subtract_P):
        return -np.dot(psi_v.T, I_subtract_P)

    def test_inequality(self, d_k, w, j_v, psi_v, I_subtract_P):
        new_v = self.value_vector + d_k * w
        new_psi = self.calc_psi(self.calc_L(new_v, True), new_v)
        left = self.calc_J(new_psi) - j_v
        right = self.alpha * w * self.calc_delta_J(psi_v, I_subtract_P)
        return left <= right
