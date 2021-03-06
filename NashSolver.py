from Util.MatrixGameSolver import *
import numpy as np
from Actions import *
import pickle
import warnings
import time

warnings.filterwarnings('ignore', '.*Ill-conditioned*')

POLICY_FILE = './data/nash_policies.pkl'
VALUE_FILE = './data/nash_values.pkl'

DEBUG = False


class NashSolver:
    def __init__(self, game):
        self.game = game
        self.alpha = 0.5  # convergence factor
        self.mu = 0.6
        self.gamma = 0.9
        self.beta = 0.9
        self.value_vector = np.zeros(game.get_n_states())  # set initial estimation to rewards

    def solve(self):
        converge_flag = False
        iteration = 0
        while not converge_flag:
            iteration += 1
            L_v, policy = self.__calc_L()
            psi_v = self.__calc_psi(L_v, self.value_vector)
            J_v = self.__calc_J(psi_v)
            if J_v <= 10e-20:
                converge_flag = True
            else:
                D_k, I_subtract_P = self.__cal_D(policy, psi_v)
                w = 1
                k = 0
                while True:
                    if self.__test_inequality(D_k, w, J_v, psi_v, I_subtract_P):
                        self.value_vector = np.add(self.value_vector, w * D_k)
                        break
                    else:
                        print('Test inequality failed, k = ', k)
                        w = self.mu * w
                        k += 1
            print('iteration ', iteration)

        f = open(POLICY_FILE , 'wb')
        pickle.dump(policy, f)
        f.close()
        f = open(VALUE_FILE , 'wb')
        pickle.dump(self.value_vector, f)
        f.close()

        return self.value_vector, policy

    def get_policy_of_all_states(self):
        policies_x = []
        for state in range(self.game.get_n_states()):
            px, py, v = self.__solve_state(state)
            policies_x.append(px)
        return policies_x

    def create_action_value_matrix(self, state, L_v, use_Lv):
        if use_Lv:
            value_vector = L_v
        else:
            value_vector = self.value_vector
        immediate_reward = self.game.get_state_reward(state)
        action_value_matrix = np.zeros((N_ACTIONS, N_ACTIONS))

        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                transition_vector = self.game.get_state_transition((i, j))[state]
                action_value_matrix[i, j] = np.dot(transition_vector, self.game.rewards) + self.gamma * np.dot(transition_vector, value_vector)
        return action_value_matrix

    def __solve_state(self, state, L_v=None, use_Lv=False):
        if self.game.is_terminal_state(state):
            return np.zeros(N_ACTIONS), np.zeros(N_ACTIONS), 0
        start = time.time()
        action_value_matrix = self.create_action_value_matrix(state, L_v, use_Lv)
        end = time.time()
        self.debug(['Create M used', end - start])
        start = time.time()
        value, policy_x, policy_y = linprog_solve(np.array(action_value_matrix))
        end = time.time()
        self.debug(['Linprog solver used', end - start])
        return policy_x, policy_y, value

    def __calc_L(self, new_v=None, use_new_v=False):
        L_v = []
        policy = []
        for state in range(self.game.get_n_states()):
            policy_x, policy_y, value = self.__solve_state(state, new_v, use_new_v)
            L_v.append(value)
            policy.append((policy_x, policy_y))
        return L_v, policy

    def __calc_psi(self, L_v, v):
        return np.subtract(L_v, v)

    def __calc_J(self, psi_v):
        return 0.5 * np.dot(psi_v.T, psi_v)

    def __cal_D(self, policy, psi_v):
        n_states = self.game.get_n_states()
        I = np.identity(n_states)
        for state in range(self.game.get_n_states()):
            policy_x = policy[state][0]
            policy_y = policy[state][1]
            for x in range(N_ACTIONS):
                for y in range(N_ACTIONS):
                    I[state] = np.subtract(I[state], self.beta * policy_x[x] * policy_y[y] *
                                           self.game.get_state_transition((x, y))[state])

        return np.dot(np.linalg.inv(I), psi_v), I

    def __calc_delta_J(self, psi_v, I_subtract_P):
        return -np.dot(psi_v.T, I_subtract_P)

    def __test_inequality(self, d_k, w, j_v, psi_v, I_subtract_P):
        new_v = self.value_vector + d_k * w
        new_l, new_policy = self.__calc_L(new_v, True)
        new_psi = self.__calc_psi(new_l, new_v)
        left = self.__calc_J(new_psi) - j_v
        right = self.alpha * w * np.dot(self.__calc_delta_J(psi_v, I_subtract_P), d_k)
        return left <= right

    def debug(self, strs):
        if DEBUG:
            line = ''
            for s in strs:
                line += str(s)
                line += ' '
            print(line)
