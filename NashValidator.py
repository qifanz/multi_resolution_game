import numpy as np
import Game
import pickle

VALUE_FILE = 'value_validator.pkl'
POLICY_FILE = 'policy.pkl'
POLICY_VALIDATOR_FILE = 'policy_validator.pkl'
class NashValidator:
    def __init__(self, game):
        self.game = game
        self.value_vector = game.rewards
        self.fixed_op_policy = self.__import_fixed_policy()
        self.gamma = 0.95

    def __import_fixed_policy(self):
        f = open(POLICY_FILE, 'rb')
        policy = pickle.load(f)
        f.close()
        fixed_policy = []
        for p in policy:
            fixed_policy.append(p[1])
        return fixed_policy

    def solve(self):
        convergence_flag = False
        iteration=0
        while not convergence_flag:
            new_value_vector = []
            for state,value in enumerate(self.value_vector):
                if self.game.is_terminal_state(state):
                    new_value_vector.append(value)
                    continue
                best_Q = -9999
                current_reward = self.game.get_state_reward(state)
                current_state_op_policy = self.fixed_op_policy[state]
                for action in range(Game.N_ACTIONS):
                    action_Q = current_reward
                    for op_action in range(Game.N_ACTIONS):
                        transition_vector = self.game.get_state_transition((action, op_action))[state]
                        for next_state in range(len(transition_vector)):
                            probability = transition_vector[next_state]
                            action_Q += self.gamma * current_state_op_policy[op_action] * probability * self.value_vector[next_state]
                    if action_Q > best_Q:
                        best_Q = action_Q
                new_value_vector.append(best_Q)
            diff = np.sum(np.subtract(np.array(new_value_vector), np.array(self.value_vector)))
            convergence_flag = diff < 10e-20
            self.value_vector = new_value_vector.copy()
            print('iteration ',iteration)
            iteration+=1
        f = open(VALUE_FILE, 'wb')
        pickle.dump(self.value_vector, f)
        f.close()
        policy = self.get_policy()

        f = open(POLICY_VALIDATOR_FILE, 'wb')
        pickle.dump(policy, f)
        f.close()
        print('finished')

    def get_policy(self):
        policy = []
        for state, value in enumerate(self.value_vector):
            if self.game.is_terminal_state(state):
                policy.append(-1)
                continue
            best_Q = -9999
            best_action = -1
            current_reward = self.game.get_state_reward(state)
            current_state_op_policy = self.fixed_op_policy[state]
            for action in range(Game.N_ACTIONS):
                action_Q = current_reward
                for op_action in range(Game.N_ACTIONS):
                    transition_vector = self.game.get_state_transition((action, op_action))[state]
                    for next_state in range(len(transition_vector)):
                        probability = transition_vector[next_state]
                        action_Q += self.gamma * current_state_op_policy[op_action] * probability * self.value_vector[
                            next_state]
                if action_Q > best_Q:
                    best_Q = action_Q
                    best_action = action
            policy.append(best_action)
        return policy

