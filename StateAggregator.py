import numpy as np
from Game import *
import math


class AggStates:
    def __init__(self, n_aggregated_states):
        self.n_aggregated_states = n_aggregated_states
        self.mapping = []
        self.states = []
        self.absorbing_state = []
        for i in range(n_aggregated_states):
            self.mapping.append([set(), set(), set()])
            self.states.append(set())

    def add_absorbing_state(self, state):
        self.absorbing_state.append(state)

    def add_boundary_state(self, aggregated_index, state):
        self.mapping[aggregated_index][1].add(state)
        self.states[aggregated_index].add(state)

    def add_interior_state(self, aggregated_index, state):
        self.mapping[aggregated_index][2].add(state)
        self.states[aggregated_index].add(state)

    def add_periphery_state(self, aggregated_index, state):
        self.mapping[aggregated_index][0].add(state)
        self.states[aggregated_index].add(state)

    def get_boundary_state(self, aggregated_index):
        return self.mapping[aggregated_index][1]

    def get_interior_state(self, aggregated_index):
        return self.mapping[aggregated_index][2]

    def get_periphery_state(self, aggregated_index):
        return self.mapping[aggregated_index][0]

    def get_all_states_in_index(self, aggregated_index):
        return self.states[aggregated_index]


class StateAggregator:
    def __init__(self, aggregate_factor=3):
        self.aggregate_factor = aggregate_factor

    def aggregate(self, game: Game):
        n_aggregated_states = self.get_AggState_count(game)
        aggregated_states = AggStates(n_aggregated_states)

        for original_state in range(game.get_n_states()):
            if game.is_terminal_state(original_state):
                aggregated_states.add_absorbing_state(original_state)
            else:
                aggregated_index = self.get_aggregate_index(game, original_state)
                is_boundary = False
                next_states = game.transition_from[original_state]
                for next_state in next_states:
                    if aggregated_index != self.get_aggregate_index(game, next_state):
                        is_boundary = True
                        aggregated_states.add_periphery_state(aggregated_index, next_state)
                if is_boundary:
                    aggregated_states.add_boundary_state(aggregated_index, original_state)
                else:
                    aggregated_states.add_interior_state(aggregated_index, original_state)
        aggregated_rewards = self.aggregate_reward(game, aggregated_states)
        aggregated_transitions = self.aggregate_transition(game, aggregated_states)
        print('finished')
        return aggregated_states, aggregated_rewards, aggregated_transitions

    def aggregate_reward(self, game, aggregated_states: AggStates):
        aggregated_rewards = []
        for aggregated_state in range(aggregated_states.n_aggregated_states):
            n_original_states = len(aggregated_states.get_boundary_state(aggregated_state)) + len(
                aggregated_states.get_interior_state(aggregated_state))
            sum_original_rewards = 0
            for original_state in aggregated_states.get_boundary_state(aggregated_state):
                sum_original_rewards += game.get_state_reward(original_state)
            for original_state in aggregated_states.get_interior_state(aggregated_state):
                sum_original_rewards += game.get_state_reward(original_state)
            aggregated_rewards.append(1 / n_original_states * sum_original_rewards)
        return aggregated_rewards

    def aggregate_transition(self, game: Game, aggregated_states: AggStates):
        whole_aggregated_transitions = {}
        for a1 in range(N_ACTIONS):
            for a2 in range(N_ACTIONS):
                whole_aggregated_transitions[(a1, a2)] = np.zeros(
                    (aggregated_states.n_aggregated_states, aggregated_states.n_aggregated_states))
                for gamma in range(aggregated_states.n_aggregated_states):
                    original_states = aggregated_states.get_all_states_in_index(gamma)
                    peripheries = aggregated_states.get_periphery_state(gamma)
                    boundaries = aggregated_states.get_boundary_state(gamma)

                    P_within = np.zeros((len(original_states), len(original_states)))
                    tmp_index_state_mapping = []
                    for i, s1 in enumerate(original_states):
                        tmp_index_state_mapping.append(s1)
                        for j, s2 in enumerate(original_states):
                            P_within[i, j] = game.get_state_transition((a1, a2))[s1, s2]

                    for gamma_prime in range(aggregated_states.n_aggregated_states):
                        if gamma_prime == gamma:
                            # by definition, P gamma gamma = 0
                            continue
                        counter = 0  # First count the number of s' in gamma' attainable from gamma
                        states_attainable = []
                        for periphery in peripheries:
                            if self.get_aggregate_index(game, periphery) == gamma_prime:
                                counter += 1
                                states_attainable.append(periphery)
                        P_out = np.zeros((len(original_states), counter))  # A matrix of size ( |s, s'| )
                        for i, s in enumerate(original_states):
                            for j, s_prime in enumerate(states_attainable):
                                P_out[i, j] = game.get_state_transition((a1, a2))[s, s_prime]

                        phi = np.dot(np.subtract(np.eye(len(original_states)), P_within).T, np.sum(P_out, axis = 1))

                        phi_boundary_sum = 0
                        for i in range(len(phi)):
                            if tmp_index_state_mapping[i] in boundaries:
                                phi_boundary_sum += phi[i]
                        whole_aggregated_transitions[(a1, a2)][gamma, gamma_prime] = phi_boundary_sum / len(boundaries)

        return whole_aggregated_transitions

    def get_aggregate_index(self, game, original_index):
        row1, col1, row2, col2 = game.state2rc(original_index)
        aggregate_row_count = math.ceil(game.n_rows / self.aggregate_factor)
        aggregate_col_count = math.ceil(game.n_cols / self.aggregate_factor)
        count = aggregate_row_count * aggregate_col_count
        new_index = count * (
                int(row1 / self.aggregate_factor) * aggregate_col_count + int(col1 / self.aggregate_factor)) + (
                            int(row2 / self.aggregate_factor) * aggregate_col_count + int(
                        col2 / self.aggregate_factor))
        return new_index

    def get_AggState_count(self, game):
        aggregate_row_count = math.ceil(game.n_rows / self.aggregate_factor)
        aggregate_col_count = math.ceil(game.n_cols / self.aggregate_factor)
        return (aggregate_col_count * aggregate_row_count) ** 2
