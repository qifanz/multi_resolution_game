import numpy as np
from Game import *
import math


class AggStates:
    def __init__(self, n_aggregated_states):
        self.n_aggregated_states = n_aggregated_states
        self.mapping = []
        for i in range(n_aggregated_states):
            self.mapping.append([set(), set(), set()])

    def add_boundary_state(self, aggregated_index, state):
        self.mapping[aggregated_index][1].add(state)

    def add_interior_state(self, aggregated_index, state):
        self.mapping[aggregated_index][2].add(state)

    def add_periphery_state(self, aggregated_index, state):
        self.mapping[aggregated_index][0].add(state)

    def get_boundary_state(self, aggregated_index):
        return self.mapping[aggregated_index][1]

    def get_interior_state(self, aggregated_index):
        return self.mapping[aggregated_index][2]

    def get_periphery_state(self, aggregated_index):
        return self.mapping[aggregated_index][0]


class StateAggregator:
    def __init__(self, aggregate_factor=9):
        self.aggregate_factor = aggregate_factor

    def aggregate(self, game: Game):
        n_aggregated_states = math.ceil(game.get_n_states() / self.aggregate_factor)
        aggregated_states = AggStates(n_aggregated_states)

        for original_state in range(game.get_n_states()):
            aggregated_index = self.__get_aggregate_index(original_state)
            is_boundary = False
            next_states = game.transition_from[original_state].keys()
            for next_state in next_states:
                if aggregated_index != self.__get_aggregate_index(next_state):
                    is_boundary = True
                    aggregated_states.add_periphery_state(aggregated_index, next_state)
            if is_boundary:
                aggregated_states.add_boundary_state(aggregated_index, original_state)
            else:
                aggregated_states.add_interior_state(aggregated_index, original_state)

        print('finished')
        return aggregated_states

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

    def aggregate_transition(self, game, aggregated_states):
        # TODO: wait for clarification
        pass

    def __get_aggregate_index(self, original_index):
        return int(original_index / self.aggregate_factor)
