import numpy as np
from AggStates import AggStates


class AggGame:
    def __init__(self, n_rows, n_cols , aggregated_states:AggStates, aggregated_rewards, aggregated_transitions):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.aggregated_states = aggregated_states
        self.states = np.arange(aggregated_states.get_total_state_count())
        self.rewards = aggregated_rewards
        self.transition = aggregated_transitions

    def is_terminal_state(self, state):
        return self.aggregated_states.is_terminal_state(state)

    def get_n_states(self):
        return self.aggregated_states.get_total_state_count()

    def get_state_reward(self, state):
        return self.rewards[state]

    def get_state_transition(self, actions):
        return self.transition[actions]