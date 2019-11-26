import numpy as np

class Game:
    def __init__(self):
        self.states = self.initialize_states()
        self.rewards = self.initialize_rewards()
        self.transitions = self.initialize_transitions()

    def get_n_states(self):
        return len(self.states)

    def get_state_reward(self, state):
        return self.rewards[state]

    def get_state_transition(self, action_pair):
        return np.array(self.transitions[action_pair[0]][action_pair[1]])

    def initialize_states(self):
        return [0, 1, 2]

    def initialize_rewards(self):
        return [1, -1, 2]

    def initialize_transitions(self):
        action_0_1 = [
            [0.1,0.8,0.1],
            [0.3,0.4,0.3],
            [0.2,0.5,0.3]
        ]
        action_1_0 = [
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3]
        ]
        action_0_0 = [
            [0.7,0.1,0.2],
            [0.3,0.3,0.4],
            [0.1,0.1,0.8]
        ]

        action_1_1 = [
            [0.2,0.1,0.7],
            [0.3,0.3,0.4],
            [0.2,0.3,0.5]
        ]
        return [
            [action_0_0, action_0_1],
            [action_1_0, action_1_1]
        ]

