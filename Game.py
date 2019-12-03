import numpy as np


class Game:
    def __init__(self):
        self.n_rows = 3
        self.n_cols = 3
        self.n_blocks = self.n_rows * self.n_cols
        self.states = self.initialize_states()
        self.crash_blocks = [(1,2)]
        self.rewards = self.initialize_rewards()
        self.transitions = self.initialize_transitions()

    def get_n_states(self):
        '''
        Get the number of states of the game
        :return: number of states
        '''
        return len(self.states)

    def get_state_reward(self, state):
        '''
        Get the reward of each state
        :param state: index of state
        :return: the reward of that state
        '''
        return self.rewards[state]

    def get_state_transition(self, action_pair):
        '''
        Get the transition matrix for pair of actions
        :param action_pair: tuple of actions (action of player1, action of player 2)
        :return: transition matrix given that pair of action
        '''
        return np.array(self.transitions[action_pair[0]][action_pair[1]])

    def __rc2state(self, row1, col1, row2, col2):
        '''
        Convert row, col of player 1 and 2 to index of state
        :param row1: row of player1
        :param col1: col of player1
        :param row2: row of player2
        :param col2: col of player2
        :return: converted index of state
        '''
        state1 = row1 * self.n_cols + col1
        state2 = row2 * self.n_cols + col2
        state = state1 * self.n_blocks + state2
        return state

    def __state2rc(self, state):
        '''
        Convert index of state to row, col of player 1 and 2
        :param state: index of state
        :return: row, col of player 1 and row, col of player 2
        '''
        state1 = state / self.n_blocks
        state2 = state % self.n_blocks
        row1 = state1 / self.n_cols
        col1 = state1 % self.n_cols
        row2 = state2 / self.n_cols
        col2 = state2 % self.n_cols
        return row1, col1, row2, col2

    def __get_crash_states(self):
        '''
        Get if a state is a crashed state (cannot be recoverd, absorbing state)
        :param state: index of state
        :return: is crashed
        '''
        return state in self.crash_states

    def initialize_states(self):
        return np.arange(0, self.n_blocks)

    def initialize_rewards(self):
        rewards = np.ones(self.n_blocks)

    def initialize_transitions(self):
        action_0_1 = [
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3]
        ]
        action_1_0 = [
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3]
        ]
        action_0_0 = [
            [0.7, 0.1, 0.2],
            [0.3, 0.3, 0.4],
            [0.1, 0.1, 0.8]
        ]

        action_1_1 = [
            [0.2, 0.1, 0.7],
            [0.3, 0.3, 0.4],
            [0.2, 0.3, 0.5]
        ]
        return [
            [action_0_0, action_0_1],
            [action_1_0, action_1_1]
        ]
