import numpy as np
from Actions import *
from math import floor

STANDARD_REWARD = 1
CRASH_REWARD = -20
END_REWARD = -20
ACTION_SUCCESSFUL_RATE = 0.8
N_ROW = 9
N_COL = 9
CRASH_BLOCKS = [(4,4)]
DEBUG = True

class Game:
    def __init__(self):
        self.n_rows = N_ROW
        self.n_cols = N_COL
        self.crash_blocks = CRASH_BLOCKS
        self.n_blocks = self.n_rows * self.n_cols

        self.states = self.__initialize_states()
        self.crash_states_p1, self.crash_states_p2 = self.__initialize_crash_states()
        self.terminal_states = self.__initialize_terminal_states()
        self.rewards = self.__initialize_rewards()
        self.transitions, self.transition_from, self.transition_to = self.__initialize_transitions()

        #self.transition_from, self.transition_to = self.__convert_transition()

        print('Game initilized')

    def is_crash_state(self, state):
        return state in self.crash_states_p1 or state in self.crash_states_p2

    def is_terminal_state(self, state):
        return state in self.terminal_states

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
        return self.transitions[(action_pair[0], action_pair[1])]

    def __convert_transition(self):
        from_mapping = []  # state -> {next_state : {action pair : prob}}, from state to next state
        to_mapping = []  # from previous state to state
        for state in range(self.get_n_states()):
            from_mapping.append({})
            to_mapping.append({})
            for i in range(N_ACTIONS):
                for j in range(N_ACTIONS):
                    from_vector = self.get_state_transition((i, j))[state]
                    to_vector = self.get_state_transition((i, j))[:, state]
                    for next_state in range(len(from_vector)):
                        probability = from_vector[next_state]
                        if probability != 0:
                            if next_state not in from_mapping[state]:
                                from_mapping[state][next_state] = {}
                            from_mapping[state][next_state][(i, j)] = probability

                    for previous_state in range(len(to_vector)):
                        probability = from_vector[previous_state]
                        if probability != 0:
                            if previous_state not in to_mapping[state]:
                                to_mapping[state][previous_state] = {}
                            to_mapping[state][previous_state][(i, j)] = probability
        return from_mapping, to_mapping

    def rc2state(self, row1, col1, row2, col2):
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

    def state2rc(self, state):
        '''
        Convert index of state to row, col of player 1 and 2
        :param state: index of state
        :return: row, col of player 1 and row, col of player 2
        '''
        state1 = floor(state / self.n_blocks)
        state2 = state % self.n_blocks
        row1 = floor(state1 / self.n_cols)
        col1 = state1 % self.n_cols
        row2 = floor(state2 / self.n_cols)
        col2 = state2 % self.n_cols
        return row1, col1, row2, col2

    def __initialize_crash_states(self):
        '''
        Get list of crash states [[p1_crash], [p2_crash]]
        :return: an array of crash states' index
        '''
        crash_states_p1 = []
        crash_states_p2 = []
        for crash_block in self.crash_blocks:
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    if (i, j) != crash_block:
                        crash_states_p2.append(self.rc2state(i, j, crash_block[0], crash_block[1]))
                        crash_states_p1.append(self.rc2state(crash_block[0], crash_block[1], i, j))
        return crash_states_p1, crash_states_p2

    def __initialize_states(self):
        return np.arange(0, self.n_blocks * self.n_blocks)

    def __initialize_rewards(self):
        rewards = np.ones(self.get_n_states()) * STANDARD_REWARD
        for crash_state in self.crash_states_p1:
            rewards[crash_state] = CRASH_REWARD
        for crash_state in self.crash_states_p2:
            rewards[crash_state] = -CRASH_REWARD
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                rewards[self.rc2state(i, j, i, j)] = END_REWARD
        return rewards

    def __initialize_terminal_states(self):
        terminal_states = []
        for crash_state in self.crash_states_p1:
            terminal_states.append(crash_state)
        for crash_state in self.crash_states_p2:
            terminal_states.append(crash_state)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                terminal_states.append(self.rc2state(i, j, i, j))
        return terminal_states

    def __initialize_transitions(self):
        whole_transition_matrix = {}
        possible_from = []  # state -> {next_state}
        possible_to = []  # from previous state to state
        for i in range(self.get_n_states()):
            possible_from.append(set())
            possible_to.append(set())

        n_states = self.get_n_states()
        for action1 in range(N_ACTIONS):
            for action2 in range(N_ACTIONS):
                transition_matrix = np.zeros((n_states, n_states))
                for i in range(n_states):
                    state_transition = self.__get_possible_transition(i, action1, action2)
                    for state_prob_tuple in state_transition:
                        next_state = int(state_prob_tuple[0]) # avoid type bug
                        prob = state_prob_tuple[1]
                        transition_matrix[i, next_state] = prob
                        if prob != 0:
                            possible_from[i].add(next_state)
                            possible_to[next_state].add(i)
                whole_transition_matrix[(action1, action2)] = transition_matrix
        return whole_transition_matrix, possible_from, possible_to

    def __get_possible_transition(self, state, action1, action2):
        '''
        For one state (pos1,pos2) and action pair, get all possible transitions and corresponding probability
        :param state: pair of state
        :param action1: action1
        :param action2:action2
        :return: list of tuples of new state and possibility.
        '''
        res = []
        row1, col1, row2, col2 = self.state2rc(state)
        single_transition_1 = self.__get_single_state_transition(row1, col1, action1)
        single_transition_2 = self.__get_single_state_transition(row2, col2, action2)
        # Start cross product to create full transition
        for transition_1 in single_transition_1:
            for transition_2 in single_transition_2:
                prob = transition_1[1] * transition_2[1]
                state = self.rc2state(transition_1[0][0], transition_1[0][1], transition_2[0][0], transition_2[0][1])
                res.append((state, prob))
        return res

    def __get_single_state_transition(self, row, col, action):
        '''
        Get the transition prob for one player using the action.
        Ex: Player@(row = 2, col =1) using action LEFT, should return
        [ ((2,0),0.9), ((2,1),0.033), ((1,1),0.033), ((2,2),0.033) ]
        :param row: row of the player
        :param col: col of the player
        :param action: action id
        :return: list of (state, prob)
        '''
        actions = [0, 1, 2, 3]
        probs = []
        actions.remove(action)
        if self.__is_action_valid(row, col, action):
            probs.append((self.__create_new_rc_from_action(row, col, action), ACTION_SUCCESSFUL_RATE))
            possible_count = 1  # at original state is always possible
            for a in actions:
                if self.__is_action_valid(row, col, a):
                    possible_count += 1
            rest_prob = (1 - ACTION_SUCCESSFUL_RATE) / possible_count
            probs.append(((row, col), rest_prob))
            for a in actions:
                if self.__is_action_valid(row, col, a):
                    probs.append((self.__create_new_rc_from_action(row, col, a), rest_prob))
        else:
            probs.append(((row, col),
                          ACTION_SUCCESSFUL_RATE))  # if action is invalid, most of the time should stay in original state
            possible_count = 0
            for a in actions:
                if self.__is_action_valid(row, col, a):
                    possible_count += 1
            rest_prob = (1 - ACTION_SUCCESSFUL_RATE) / possible_count
            for a in actions:
                if self.__is_action_valid(row, col, a):
                    probs.append((self.__create_new_rc_from_action(row, col, a), rest_prob))

        return probs

    def __create_new_rc_from_action(self, row, col, action):
        return row + get_movement(action)[0], col + get_movement(action)[1]

    def __is_action_valid(self, row, col, action):
        new_row = row + get_movement(action)[0]
        new_col = col + get_movement(action)[1]
        if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
            return True
        return False

    def get_success_next_state(self, state, action1, action2):
        '''
        return next state if action 1 and action 2 BOTH SUCCEED
        :param state: current state
        :param action1: action executed by p1
        :param action2: action executed by p2
        :return: next state if action1 and action2 both succeed
        '''
        row1, col1, row2, col2 = self.state2rc(state)
        if self.__is_action_valid(row1, col1, action1):
            new_pos1 = self.__create_new_rc_from_action(row1, col1, action1)
        else:
            new_pos1 = row1, col1
        if self.__is_action_valid(row2, col2, action2):
            new_pos2 = self.__create_new_rc_from_action(row2, col2, action2)
        else:
            new_pos2 = row2, col2
        return self.rc2state(new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1])



    def print_state(self, state):
        row1, col1, row2, col2 = self.state2rc(state)
        print('----------------')
        for i in range(self.n_rows):
            line = '|'
            for j in range(self.n_cols):
                if i == row1 == row2 and j == col1 == col2:
                    line += (' O |')
                else:
                    if (i, j) in self.crash_blocks:
                        line += ' C |'
                    elif i == row1 and j == col1:
                        line += (' X |')
                    elif i == row2 and j == col2:
                        line += (' Y |')
                    else:
                        line += ('   |')
            print(line)
            #print('----------------')

        mat = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i == row1 == row2 and j == col1 == col2:
                    mat[i, j] = 3
                else:
                    if (i, j) in self.crash_blocks:
                        mat[i, j] = -1
                    elif i == row1 and j == col1:
                        mat[i, j] = 1
                    elif i == row2 and j == col2:
                        mat[i, j] = 2

        return mat
