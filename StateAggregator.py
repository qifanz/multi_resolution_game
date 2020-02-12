import math

from AggGame import AggGame
from AggStates import AggStates
from Game import *


class StateAggregator:
    def __init__(self, game: Game, aggregate_factor=3):
        self.game = game
        self.aggregate_factor = aggregate_factor
        self.n_absorbing_type = game.n_absorbing_type

    def aggregate(self):
        n_aggregated_states = self.get_AggState_count()
        aggregated_states = AggStates(n_aggregated_states, self.n_absorbing_type)

        for original_state in range(self.game.get_n_states()):
            if self.game.is_terminal_state(original_state):
                aggregated_states.add_absorbing_state(self.game.get_state_absorbing_type(original_state),
                                                      original_state)
            else:
                aggregated_index = self.get_aggregate_index(original_state)
                is_boundary = False
                next_states = self.game.transition_from[original_state]
                for next_state in next_states:
                    if aggregated_index != self.get_aggregate_index(next_state):
                        is_boundary = True
                        aggregated_states.add_periphery_state(aggregated_index, next_state)
                if is_boundary:
                    aggregated_states.add_boundary_state(aggregated_index, original_state)
                else:
                    aggregated_states.add_interior_state(aggregated_index, original_state)
        aggregated_rewards = self.aggregate_reward(aggregated_states)
        aggregated_transitions = self.aggregate_transition(aggregated_states)
        print('finished')

        aggregate_row_count = math.ceil(self.game.n_rows / self.aggregate_factor)
        aggregate_col_count = math.ceil(self.game.n_cols / self.aggregate_factor)
        return AggGame(aggregate_row_count, aggregate_col_count, aggregated_states, aggregated_rewards, aggregated_transitions)

    def aggregate_reward(self, aggregated_states: AggStates):
        aggregated_rewards = []
        for aggregated_state in range(aggregated_states.n_aggregated_states):
            n_original_states = len(aggregated_states.get_boundary_state(aggregated_state)) + len(
                aggregated_states.get_interior_state(aggregated_state))
            sum_original_rewards = 0
            for original_state in aggregated_states.get_boundary_state(aggregated_state):
                sum_original_rewards += self.game.get_state_reward(original_state)
            for original_state in aggregated_states.get_interior_state(aggregated_state):
                sum_original_rewards += self.game.get_state_reward(original_state)
            aggregated_rewards.append(1 / n_original_states * sum_original_rewards)
        for aggregated_absorbing_state in range(aggregated_states.n_absorbing_type):
            aggregated_rewards.append(self.game.get_absorbing_type_reward(aggregated_absorbing_state))
        return aggregated_rewards

    def aggregate_transition(self, aggregated_states: AggStates):
        whole_aggregated_transitions = {}
        for a1 in range(N_ACTIONS):
            for a2 in range(N_ACTIONS):
                whole_aggregated_transitions[(a1, a2)] = np.zeros(
                    (aggregated_states.get_total_state_count(), aggregated_states.get_total_state_count()))
                # Plus one since we have an aggregated state for absorbing states
                for gamma in range(aggregated_states.n_aggregated_states):
                    original_states = aggregated_states.get_all_states_in_index(gamma)
                    peripheries = aggregated_states.get_periphery_state(gamma)
                    boundaries = aggregated_states.get_boundary_state(gamma)

                    P_within, tmp_index_state_mapping = self.calc_P_within(a1, a2, original_states)

                    for gamma_prime in range(aggregated_states.n_aggregated_states):
                        if gamma_prime == gamma:  # by definition, P gamma gamma = 0
                            continue

                        P_out = self.calc_P_out(a1, a2, gamma_prime, original_states, peripheries)
                        phi = np.dot(np.linalg.inv(np.subtract(np.eye(len(original_states)), P_within)),
                                     np.sum(P_out, axis=1))
                        phi_boundary_sum = 0
                        for i in range(len(phi)):
                            if tmp_index_state_mapping[i] in boundaries:
                                phi_boundary_sum += phi[i]
                        whole_aggregated_transitions[(a1, a2)][gamma, gamma_prime] = phi_boundary_sum / len(boundaries)

                    for gamma_absorbing in range(aggregated_states.n_absorbing_type):
                        P_out_absorbing = self.calc_P_out_absorb(a1, a2, original_states, peripheries, gamma_absorbing)
                        phi = np.dot(np.linalg.inv(np.subtract(np.eye(len(original_states)), P_within)),
                                     np.sum(P_out_absorbing, axis=1))
                        phi_boundary_sum = 0
                        for i in range(len(phi)):
                            if tmp_index_state_mapping[i] in boundaries:
                                phi_boundary_sum += phi[i]
                        whole_aggregated_transitions[(a1, a2)][
                            gamma, aggregated_states.n_aggregated_states + gamma_absorbing] = phi_boundary_sum / len(
                            boundaries)

        return whole_aggregated_transitions

    def calc_P_within(self, a1, a2, original_states):
        P_within = np.zeros((len(original_states), len(original_states)))
        tmp_index_state_mapping = []
        for i, s1 in enumerate(original_states):
            tmp_index_state_mapping.append(s1)
            for j, s2 in enumerate(original_states):
                P_within[i, j] = self.game.get_state_transition((a1, a2))[s1, s2]
        return P_within, tmp_index_state_mapping

    def calc_P_out(self, a1, a2, gamma_prime, original_states, peripheries):
        counter = 0  # First count the number of s' in gamma' attainable from gamma
        states_attainable = []
        for periphery in peripheries:
            if self.get_aggregate_index(periphery) == gamma_prime:
                counter += 1
                states_attainable.append(periphery)
        P_out = np.zeros((len(original_states), counter))  # A matrix of size ( |s, s'| )
        for i, s in enumerate(original_states):
            for j, s_prime in enumerate(states_attainable):
                P_out[i, j] = self.game.get_state_transition((a1, a2))[s, s_prime]
        return P_out

    def calc_P_out_absorb(self, a1, a2, original_states, peripheries, absorbing_type):
        states_attainable = []
        counter = 0
        for periphery in peripheries:
            if self.game.is_terminal_state(periphery):
                if self.game.get_state_absorbing_type(periphery) == absorbing_type:
                    counter += 1
                    states_attainable.append(periphery)
        P_out = np.zeros((len(original_states), counter))  # A matrix of size ( |s, s'| )
        for i, s in enumerate(original_states):
            for j, s_prime in enumerate(states_attainable):
                P_out[i, j] = self.game.get_state_transition((a1, a2))[s, s_prime]
        return P_out

    def get_aggregate_index(self, original_index):
        if self.game.is_terminal_state(original_index):
            return self.get_AggState_count() + self.game.get_state_absorbing_type(original_index)
        row1, col1, row2, col2 = self.game.state2rc(original_index)
        aggregate_row_count = math.ceil(self.game.n_rows / self.aggregate_factor)
        aggregate_col_count = math.ceil(self.game.n_cols / self.aggregate_factor)
        count = aggregate_row_count * aggregate_col_count
        new_index = count * (
                int(row1 / self.aggregate_factor) * aggregate_col_count + int(col1 / self.aggregate_factor)) + (
                            int(row2 / self.aggregate_factor) * aggregate_col_count + int(
                        col2 / self.aggregate_factor))
        return new_index

    def belong_to_same_AggState(self, s1, s2):
        return self.get_aggregate_index(s1) == self.get_aggregate_index(s2)

    def get_AggState_count(self):
        aggregate_row_count = math.ceil(self.game.n_rows / self.aggregate_factor)
        aggregate_col_count = math.ceil(self.game.n_cols / self.aggregate_factor)
        return (aggregate_col_count * aggregate_row_count) ** 2
